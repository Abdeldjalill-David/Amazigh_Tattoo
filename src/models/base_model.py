"""
Base model loader optimized for 24GB VRAM.
Handles Stable Diffusion 1.5, SDXL, and custom checkpoints.
"""

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AmazighBaseModel:
    """
    High-performance base model loader with 24GB VRAM optimizations.
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enable_xformers: bool = True,
        use_sdxl: bool = False
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.use_sdxl = use_sdxl
        self.pipe = None
        self.vae = None
        self.unet = None
        
        # VRAM tracking for 24GB optimization
        self.vram_stats = {
            'total_gb': 24,
            'reserved_gb': 0,
            'allocated_gb': 0
        }
        
    def load(self, custom_vae: Optional[str] = None) -> StableDiffusionPipeline:
        """
        Load model with 24GB VRAM optimizations.
        """
        logger.info(f"Loading {'SDXL' if self.use_sdxl else 'SD 1.5'} model...")
        logger.info(f"Device: {self.device} | Dtype: {self.dtype}")
        
        # Model loading kwargs optimized for 24GB
        load_kwargs = {
            "torch_dtype": self.dtype,
            "use_safetensors": True,
            "safety_checker": None,  # Disable for speed
            "requires_safety_checker": False
        }
        
        # Add variant for SDXL fp16
        if self.use_sdxl and self.dtype == torch.float16:
            load_kwargs["variant"] = "fp16"
            
        # Load pipeline
        pipe_class = StableDiffusionXLPipeline if self.use_sdxl else StableDiffusionPipeline
        self.pipe = pipe_class.from_pretrained(self.model_id, **load_kwargs)
        
        # Custom VAE for better quality (optional)
        if custom_vae:
            from diffusers import AutoencoderKL
            self.pipe.vae = AutoencoderKL.from_pretrained(
                custom_vae,
                torch_dtype=self.dtype
            )
        
        # Move to device
        self.pipe = self.pipe.to(self.device)
        
        # 24GB VRAM Optimizations
        self._apply_memory_optimizations()
        
        # Store references
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        
        self._log_vram_usage()
        logger.info("✅ Model loaded successfully!")
        
        return self.pipe
    
    def _apply_memory_optimizations(self):
        """Apply optimizations suitable for 24GB VRAM."""
        
        # Enable xformers for memory-efficient attention (saves ~2-3GB)
        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("✅ XFormers enabled")
            except Exception as e:
                logger.warning(f"XFormers not available: {e}")
        
        # VAE optimizations for large images
        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()
            logger.info("✅ VAE slicing enabled")
            
        if hasattr(self.pipe, 'enable_vae_tiling') and self.use_sdxl:
            self.pipe.enable_vae_tiling()
            logger.info("✅ VAE tiling enabled")
        
        # For 24GB: We can keep model on GPU, but enable sequential CPU offload
        # only if doing very large batch processing
        # self.pipe.enable_sequential_cpu_offload()  # Uncomment if needed
        
        # Compile UNet for speed (PyTorch 2.0+)
        if hasattr(torch, 'compile') and torch.cuda.get_device_capability()[0] >= 8:
            logger.info("Compiling UNet with torch.compile...")
            self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
    
    def _log_vram_usage(self):
        """Track VRAM usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"VRAM Usage: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            self.vram_stats['allocated_gb'] = allocated
            self.vram_stats['reserved_gb'] = reserved
    
    def get_optimal_batch_size(self, resolution: int = 512) -> int:
        """
        Calculate optimal batch size for 24GB VRAM.
        """
        # Rough heuristic for 24GB
        if resolution <= 512:
            return 8 if not self.use_sdxl else 4
        elif resolution <= 768:
            return 4 if not self.use_sdxl else 2
        elif resolution <= 1024:
            return 2 if not self.use_sdxl else 1
        return 1
    
    def set_scheduler(self, scheduler_type: str = "dpmsolver++"):
        """Set optimized scheduler for fast inference."""
        if scheduler_type == "dpmsolver++":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True
            )
        elif scheduler_type == "euler":
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )