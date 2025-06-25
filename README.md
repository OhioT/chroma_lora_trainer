# Chroma LoRA Training

This project provides a simple pipeline for training LoRA weights on the Chroma diffusion model. The workflow includes embedding caching, LoRA training, and image generation. The aim is to minimize the number of options and config files. If you have a <24GB GPU or complex requirements, try https://github.com/ostris/ai-toolkit instead.

## Installation

### Prerequisites
- 24GB Nvidia GPU
- Python 3.10
- [uv](https://github.com/astral-sh/uv) recommended (or venv)

### Setup Environment with uv

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/OhioT/chroma_lora_trainer
cd chroma_lora_trainer
```

## Workflow Overview

The complete training pipeline consists of 4 main steps:

1. **Dataset Preparation** - Organize images and captions
2. **Embedding Caching** - Process and cache embeddings for efficient training
3. **LoRA Training** - Train LoRA weights on the cached data
4. **Image Generation** - Sample images using the trained LoRA
5. **ComfyUI Use** - Copy the Lora to the ComfyUI/models/lora folder for use with Chroma in ComfyUI

## Step 1: Dataset Preparation

### Folder Structure
Create your dataset in the following structure:
```
tags/
└── your_tag_name/
    ├── image1.jpg
    ├── image1.txt
    ├── image2.png
    ├── image2.txt
    └── ...
```

- **Images**: Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`
- **Captions**: Text files with the same basename as images but with `.txt` extension
- **Content**: Each `.txt` file should contain a caption describing the corresponding image

## Step 2: Embedding Caching

Cache the embeddings for your dataset and sample prompt:

```bash
uv run cache.py \
    --image_folder_path path/to/your/images \
    --base_resolution 512 \
    --prompt "your sample prompt for testing"
```

### Parameters
- `--image_folder_path`: Path to your images and captions
- `--base_resolution`: Base resolution for aspect ratio buckets (512 or 1024)
- `--prefix_tag`: Prepends the tag name to each caption during training
- `--prompt`: Sample prompt to cache for testing during training
- `--negative_prompt`: Negative prompt for sampling

### What happens
- Processes all images in `tags/your_tag_name/`
- Encodes images using VAE and text using T5 encoder
- Saves cached embeddings to `tags/your_tag_name/embeddings/`
- Caches sample prompt to `tags/your_tag_name/embeddings/sample_prompt.pkl`

## Step 3: LoRA Training

Train LoRA weights using the cached embeddings.
Review the samples every N epochs and stop training when they look good. You do not need to wait for all the epochs to finish.

```bash
uv run train.py \
    --name your_lora_name \
    --image_folder_path path/to/your/images \
    --model_path chroma-unlocked-v39.safetensors \
    --lora_rank 16 \
    --lora_alpha 16 \
    --learning_rate 1e-4 \
    --save_every_n_epochs 10 \
    --sample_every_n_epochs 10
```

### Key Parameters
- `--name`: The name of your Lora
- `--image_folder_path`: Path to your images and captions (now with cached embeddings)
- `--model_path`: Path to base Chroma model file
- `--lora_rank`: LoRA rank (lower = fewer parameters, faster training)
- `--lora_alpha`: LoRA alpha (scaling factor)
- `--learning_rate`: Learning rate for training
- `--save_every_n_epochs`: Save checkpoint frequency
- `--sample_every_n_epochs`: Generate sample image frequency

### Training Output
- **LoRA weights**: Saved to `lora/name/lora_name_step_XXXX.safetensors` every N epochs
- **Sample images**: Saved to `lora/name/samples/`  every N epochs

## Step 4: Image Generation

Generate images using your trained LoRA:

```bash

uv run cache.py --prompt "your sample prompt for testing" --output my_sample_prompt.pkl

uv run sample.py \
    --model_path chroma-unlocked-v39.safetensors \
    --lora_path lora/name/lora_name_step_1000.safetensors \
    --prompt my_sample_prompt.pkl \
    --output generated_image.png \
    --width 512 \
    --height 512 \
    --steps 25 \
    --guidance_scale 4.0 \
    --seed 42
```

### Parameters
- `--model_path`: Path to base Chroma model
- `--lora_path`: Path to trained LoRA weights (optional)
- `--prompt`: Path to cached prompt embeddings
- `--output`: Output image filename
- `--width/height`: Generated image dimensions
- `--steps`: Number of diffusion steps
- `--guidance_scale`: CFG strength
- `--seed`: Random seed for reproducible results
