#!/bin/bash
# Setup script for Amazigh Tattoo Generator on 24GB GPU

echo "🎨 Setting up Amazigh Tattoo Generator..."
echo "Detected GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Create directories
echo "Creating project structure..."
mkdir -p data/{raw,processed/{train,validation},captions,metadata}
mkdir -p outputs/{checkpoints,samples,logs,exports}
mkdir -p configs src/{models,data_processing,training,generation,utils} notebooks scripts tests docs

# Install dependencies
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q diffusers transformers accelerate peft xformers
pip install -q bitsandbytes  # For 8-bit optimizers (optional with 24GB)
pip install -q pillow opencv-python matplotlib seaborn
pip install -q pyyaml tqdm wandb tensorboard
pip install -q jupyterlab ipywidgets
pip install -q safetensors huggingface-hub
pip install -q datasets  # HuggingFace datasets

# Verify xformers installation
python -c "import xformers; print('✅ XFormers installed')"

# Download base models (optional - can be done later)
echo "Models will be downloaded on first use or during training"

echo "✅ Setup complete! Activate your environment and start with:"
echo "   python -m src.training.train_lora"