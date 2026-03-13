"""
LoRA fine-tuning implementation optimized for 24GB VRAM.
Supports custom Amazigh tattoo datasets with high-quality results.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import CLIPTokenizer
from PIL import Image
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import wandb  # Optional: for experiment tracking

logger = logging.getLogger(__name__)


class AmazighTattooDataset(Dataset):
    """
    Custom dataset for Amazigh tattoos with rich metadata.
    """
    
    def __init__(
        self,
        data_root: str,
        tokenizer: CLIPTokenizer,
        resolution: int = 512,
        center_crop: bool = True,
        flip_p: float = 0.5
    ):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.center_crop = center_crop
        self.flip_p = flip_p
        
        # Load image-caption pairs
        self.data = self._load_data()
        logger.info(f"Loaded {len(self.data)} training samples")
    
    def _load_data(self) -> List[Dict]:
        """Load images with their captions."""
        data = []
        image_dir = self.data_root / "processed" / "train"
        caption_dir = self.data_root / "captions"
        
        for img_path in image_dir.glob("*.png") or image_dir.glob("*.jpg"):
            caption_path = caption_dir / f"{img_path.stem}.txt"
            if caption_path.exists():
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                
                # Load metadata if available
                meta_path = self.data_root / "metadata" / f"{img_path.stem}.json"
                metadata = {}
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                
                data.append({
                    'image': str(img_path),
                    'caption': caption,
                    'metadata': metadata
                })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and transform image
        image = Image.open(item['image']).convert('RGB')
        image = image.resize((self.resolution, self.resolution))
        
        # Tokenize caption
        tokens = self.tokenizer(
            item['caption'],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Convert to tensor
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5])  # Scale to [-1, 1]
        ])
        
        return {
            'pixel_values': transform(image),
            'input_ids': tokens.input_ids.squeeze(),
            'caption': item['caption'],
            'metadata': item['metadata']
        }


class AmazighLoRATrainer:
    """
    Professional LoRA trainer for Amazigh tattoo generation.
    Optimized for 24GB VRAM with advanced features.
    """
    
    def __init__(
        self,
        config_path: str = "configs/model_config.yaml",
        output_dir: str = "outputs/checkpoints"
    ):
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.lora_model = None
        self.optimizer = None
        self.lr_scheduler = None
        
    def _load_config(self, path: str) -> Dict:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_model(self, base_model: 'AmazighBaseModel'):
        """Configure model for LoRA training."""
        logger.info("Setting up LoRA configuration...")
        
        config = self.config['model']['lora']
        
        # LoRA config for high-quality adaptation
        lora_config = LoraConfig(
            r=config['rank'],                    # 64 for 24GB
            lora_alpha=config['alpha'],          # 64
            target_modules=config['target_modules'],
            lora_dropout=config['dropout'],
            bias=config['bias'],
            modules_to_save=["conv_in", "conv_out"]  # Save input/output convs
        )
        
        # Apply LoRA to UNet
        self.lora_model = get_peft_model(base_model.unet, lora_config)
        
        # Optionally train text encoder for better prompt understanding
        if self.config['model']['training']['train_text_encoder']:
            text_lora_config = LoraConfig(
                r=16,  # Lower rank for text encoder
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05
            )
            base_model.pipe.text_encoder = get_peft_model(
                base_model.pipe.text_encoder, 
                text_lora_config
            )
        
        # Enable gradient checkpointing for memory efficiency
        if self.config['model']['vram']['gradient_checkpointing']:
            self.lora_model.enable_gradient_checkpointing()
            if hasattr(base_model.pipe.text_encoder, 'gradient_checkpointing_enable'):
                base_model.pipe.text_encoder.gradient_checkpointing_enable()
        
        self.model = base_model
        logger.info(f"LoRA parameters: {self.lora_model.print_trainable_parameters()}")
        
        return self.lora_model
    
    def setup_training(self, dataset: AmazighTattooDataset):
        """Configure optimizer and dataloader."""
        train_config = self.config['model']['training']
        
        # DataLoader with optimal settings for 24GB
        self.dataloader = DataLoader(
            dataset,
            batch_size=train_config['batch_size'],  # 4 for 24GB
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        # Optimizer - AdamW with 8-bit for memory (optional with 24GB)
        from torch.optim import AdamW
        
        params_to_optimize = list(self.lora_model.parameters())
        if self.config['model']['training']['train_text_encoder']:
            params_to_optimize += list(self.model.pipe.text_encoder.parameters())
        
        self.optimizer = AdamW(
            params_to_optimize,
            lr=train_config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Learning rate scheduler
        from transformers import get_cosine_schedule_with_warmup
        
        total_steps = len(self.dataloader) * train_config['num_epochs']
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=train_config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.model.model_id,
            subfolder="scheduler"
        )
        
        logger.info(f"Training setup complete. Total steps: {total_steps}")
    
    def train(self, num_epochs: Optional[int] = None):
        """Main training loop with advanced features."""
        if num_epochs is None:
            num_epochs = self.config['model']['training']['num_epochs']
        
        # Initialize wandb for tracking (optional)
        try:
            wandb.init(project="amazigh-tattoo-lora", config=self.config)
        except:
            logger.info("Wandb not initialized, continuing without tracking")
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        global_step = 0
        self.model.unet.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            with tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for step, batch in enumerate(pbar):
                    # Move to device
                    pixel_values = batch['pixel_values'].to(self.device, dtype=self.model.dtype)
                    input_ids = batch['input_ids'].to(self.device)
                    
                    # Encode images to latent space
                    with torch.no_grad():
                        latents = self.model.vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * self.model.vae.config.scaling_factor
                    
                    # Add noise
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps,
                        (latents.size(0),), device=self.device
                    ).long()
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # Get text embeddings
                    encoder_hidden_states = self.model.pipe.text_encoder(input_ids)[0]
                    
                    # Predict noise
                    noise_pred = self.model.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states
                    ).sample
                    
                    # Loss calculation
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    
                    # Backprop
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.lora_model.parameters(),
                        self.config['model']['training']['max_grad_norm']
                    )
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Logging
                    epoch_loss += loss.item()
                    global_step += 1
                    
                    pbar.set_postfix({'loss': loss.item(), 'lr': self.lr_scheduler.get_last_lr()[0]})
                    
                    # Save checkpoint
                    if global_step % self.config['model']['training']['save_steps'] == 0:
                        self.save_checkpoint(global_step)
                    
                    # Log to wandb
                    if global_step % 50 == 0:
                        try:
                            wandb.log({
                                "loss": loss.item(),
                                "lr": self.lr_scheduler.get_last_lr()[0],
                                "epoch": epoch
                            })
                        except:
                            pass
            
            avg_loss = epoch_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Final save
        self.save_checkpoint("final", save_full_model=True)
        logger.info("Training completed!")
    
    def save_checkpoint(self, step: int, save_full_model: bool = False):
        """Save LoRA weights."""
        save_path = self.output_dir / f"amazigh_lora_step_{step}"
        save_path.mkdir(exist_ok=True)
        
        # Save LoRA weights
        self.lora_model.save_pretrained(save_path / "unet_lora")
        
        if self.config['model']['training']['train_text_encoder']:
            self.model.pipe.text_encoder.save_pretrained(save_path / "text_encoder_lora")
        
        # Save config
        import json
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Checkpoint saved to {save_path}")