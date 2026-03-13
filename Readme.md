# 🎨 Amazigh Tattoo Generator (Yirawen)

Professional AI system for generating authentic Amazigh (Berber) traditional tattoos, optimized for 24GB VRAM GPUs.

## Project Structure
amazigh-tattoo-ai/
├── configs/          # YAML configurations
├── data/            # Dataset management
├── src/             # Modular source code
├── notebooks/       # Exploration & demos
├── scripts/         # Automation scripts
├── outputs/         # Results (gitignored)
└── tests/           # Unit tests



## Quick Start

```bash
# 1. Setup
bash scripts/setup_environment.sh

# 2. Prepare data (see data/README.md for collection guide)
python -m src.data_processing.dataset_builder

# 3. Train LoRA (optimized for 24GB VRAM)
python -m src.training.train_lora --config configs/model_config.yaml

# 4. Generate
python -c "from src.models.base_model import AmazighBaseModel; ..."