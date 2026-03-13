"""
High-performance tattoo generation with advanced prompting.
"""

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFilter
import random
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class AmazighTattooGenerator:
    """
    Professional tattoo generator with cultural authenticity.
    """
    
    # Amazigh symbol library for structured generation
    SYMBOL_LIBRARY = {
        'protection': ['fibula', 'hand_of_fatima', 'khamsa', 'eye'],
        'fertility': ['partridge_foot', 'ram_horns', 'tree_of_life'],
        'beauty': ['sun', 'moon', 'diamond', 'chevron'],
        'identity': ['tazerzit', 'cross', 'dots_pattern', 'line_pattern'],
        'healing': ['snake', 'scorpion', 'spider']
    }
    
    LOCATION_STYLES = {
        'chin': 'vertical geometric lines, symmetrical, bold',
        'forehead': 'central symbol, radiating patterns, elegant',
        'cheek': 'angular patterns, warrior marks, strong lines',
        'hand': 'intricate web, connecting lines, detailed',
        'wrist': 'band pattern, wrapping, continuous',
        'arm': 'flowing design, storytelling, sequential'
    }
    
    def __init__(
        self,
        base_model: 'AmazighBaseModel',
        lora_path: Optional[str] = None,
        device: str = "cuda"
    ):
        self.base_model = base_model
        self.device = device
        self.pipe = base_model.pipe
        
        # Load LoRA if provided
        if lora_path:
            self.load_lora(lora_path)
        
        # Set optimal scheduler for quality
        self.base_model.set_scheduler("dpmsolver++")
        
        # Generation tracking
        self.generation_history = []
    
    def load_lora(self, lora_path: str, alpha: float = 1.0):
        """Load fine-tuned LoRA weights."""
        logger.info(f"Loading LoRA from {lora_path}...")
        
        self.pipe.unet = PeftModel.from_pretrained(
            self.pipe.unet,
            lora_path,
            adapter_name="amazigh_tattoo"
        )
        self.pipe.unet.set_adapter("amazigh_tattoo")
        self.pipe.unet.merge_and_unload()  # Merge for faster inference
        
        logger.info("LoRA loaded successfully")
    
    def generate(
        self,
        description: str,
        style: str = "traditional",
        location: str = "chin",
        num_images: int = 4,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        size: Tuple[int, int] = (512, 512),
        add_symbol_meaning: bool = False
    ) -> List[Image.Image]:
        """
        Generate authentic Amazigh tattoo designs.
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        
        # Build culturally-informed prompt
        prompt = self._build_prompt(description, style, location, add_symbol_meaning)
        negative_prompt = self._build_negative_prompt()
        
        logger.info(f"Generating: {description} | Prompt: {prompt[:100]}...")
        
        # Generate with optimal settings for 24GB VRAM
        with torch.inference_mode():
            images = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                height=size[1],
                width=size[0],
                generator=torch.Generator(self.device).manual_seed(seed) if seed else None
            ).images
        
        # Post-process for tattoo clarity
        processed_images = [self._post_process(img) for img in images]
        
        # Log generation
        self.generation_history.append({
            'prompt': prompt,
            'style': style,
            'location': location,
            'seed': seed
        })
        
        return processed_images
    
    def _build_prompt(
        self, 
        description: str, 
        style: str, 
        location: str,
        add_meaning: bool
    ) -> str:
        """Construct culturally authentic prompt."""
        
        # Base cultural context
        base = "Traditional Amazigh Berber tattoo, Yirawen, "
        
        # Location-specific styling
        location_style = self.LOCATION_STYLES.get(location, "traditional placement")
        
        # Style modifiers
        style_mods = {
            'traditional': 'authentic, hand-poked, ancient technique',
            'modern': 'contemporary interpretation, clean lines',
            'minimalist': 'simple, reduced, essential lines',
            'ornate': 'detailed, elaborate, decorative'
        }
        
        # Construct
        prompt = f"{base}{description}, {location} placement, {location_style}, "
        prompt += f"{style_mods.get(style, style)}, "
        prompt += "black ink on skin, geometric patterns, lines and dots, "
        prompt += "high contrast, clean line art, white background, "
        prompt += "anthropological accuracy, North African heritage"
        
        if add_meaning:
            # Add symbolic meaning context
            prompt += ", symbolic protection marks, cultural significance"
        
        return prompt
    
    def _build_negative_prompt(self) -> str:
        """Build negative prompt to avoid common issues."""
        return """
        colorful, multiple colors, red, blue, green, 
        modern western style, tribal tattoo, Maori, Polynesian,
        photorealistic skin texture, 3d render, shadow, 
        blurry, low quality, distorted, deformed,
        gradient fill, soft edges, watercolor, brush strokes,
        text, watermark, signature, frame, border
        """
    
    def _post_process(self, image: Image.Image) -> Image.Image:
        """Enhance image for tattoo clarity."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance contrast for black ink
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Sharpen lines
        image = image.filter(ImageFilter.SHARPEN)
        
        return image
    
    def batch_generate_variations(
        self,
        base_description: str,
        locations: List[str] = None,
        styles: List[str] = None,
        num_per_combo: int = 2
    ) -> Dict[str, List[Image.Image]]:
        """
        Generate systematic variations for dataset building.
        """
        if locations is None:
            locations = ['chin', 'forehead', 'hand']
        if styles is None:
            styles = ['traditional', 'minimalist']
        
        results = {}
        
        for location in locations:
            for style in styles:
                key = f"{base_description}_{location}_{style}"
                images = self.generate(
                    description=base_description,
                    style=style,
                    location=location,
                    num_images=num_per_combo
                )
                results[key] = images
        
        return results
    
    def generate_symbol_set(
        self,
        symbol_type: str,
        num_designs: int = 10
    ) -> List[Image.Image]:
        """
        Generate variations of a specific symbol type.
        """
        symbols = self.SYMBOL_LIBRARY.get(symbol_type, ['geometric'])
        images = []
        
        for symbol in symbols:
            desc = f"{symbol} symbol, traditional pattern"
            imgs = self.generate(desc, num_images=num_designs // len(symbols))
            images.extend(imgs)
        
        return images