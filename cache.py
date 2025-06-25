import os
import torch
import argparse
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
import hashlib
from datasets import Dataset
import torchvision.transforms as transforms
import pickle

from pipeline import load_vae, TextEncoder, aspect_ratio_buckets_512, aspect_ratio_buckets_1024

def cache_prompt_embedding(
    text_encoder: TextEncoder,
    prompt: str,
    negative_prompt: str = "",
    output_path: str = "cached_prompt.pkl"
):

    print(f"Processing positive prompt: '{prompt}'")
    positive_embeds, positive_mask, _ = text_encoder.encode_text(prompt)
    

    print(f"Processing negative prompt: '{negative_prompt}'")
    negative_embeds, negative_mask, _ = text_encoder.encode_text(negative_prompt)
    
    cached_data = {
        'positive': {
            'text_embeds': positive_embeds,
            'attention_mask': positive_mask,
            'prompt_text': prompt
        },
        'negative': {
            'text_embeds': negative_embeds,
            'attention_mask': negative_mask,
            'prompt_text': negative_prompt
        },
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(cached_data, f)
    
    print(f"Cached embeddings saved to: {output_path}")
    
    return cached_data

def find_nearest_bucket(aspect_ratio_buckets: List[Tuple[int, int]], image_width: int, image_height: int) -> Tuple[int, int]:
    image_aspect_ratio = image_width / image_height
    
    best_bucket = aspect_ratio_buckets[0]
    best_diff = float('inf')
    
    for bucket_width, bucket_height in aspect_ratio_buckets:
        bucket_aspect_ratio = bucket_width / bucket_height
        diff = abs(image_aspect_ratio - bucket_aspect_ratio)
        
        if diff < best_diff:
            best_diff = diff
            best_bucket = (bucket_width, bucket_height)
    
    return best_bucket

def encode_image(vae, aspect_ratio_buckets: List[Tuple[int, int]], image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Encode image using VAE with appropriate aspect ratio bucket"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    bucket_width, bucket_height = find_nearest_bucket(aspect_ratio_buckets, image.width, image.height)
    target_size = (bucket_height, bucket_width)
    print(target_size)
    
    image_transform = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) 
    ])
    
    image_tensor = image_transform(image).unsqueeze(0).to(
        vae.device, dtype=vae.dtype
    )
    
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
    return latents, (bucket_width, bucket_height)

def process_dataset_folder(
    vae,
    aspect_ratio_buckets: List[Tuple[int, int]],
    text_encoder: TextEncoder,
    folder_path: str, 
    caption_ext: str = "txt",
    supported_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.webp']
) -> List[Dict]:

    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError(f"Dataset folder does not exist: {folder_path}")
        
    image_files = []
    for ext in supported_extensions:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        raise ValueError(f"No image files found in {folder_path}")
        
    print(f"Found {len(image_files)} images in {folder_path}")
    
    cached_data = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            image = Image.open(image_path)
            
            caption_path = image_path.with_suffix(f'.{caption_ext}')

            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()

            text_embeds, attention_mask, _ = text_encoder.encode_text([caption])
            image_latents, bucket_size = encode_image(vae, aspect_ratio_buckets, image)
            
            file_id = hashlib.md5(str(image_path).encode()).hexdigest()
            
            cached_data.append({
                'id': file_id,
                'image_path': str(image_path),
                'caption': caption,
                'text_embeds': text_embeds.cpu(),
                'attention_mask': attention_mask.cpu(),
                'image_latents': image_latents.cpu(),
                'image_size': list(image.size), 
                'bucket_size': list(bucket_size), 
                'latent_size': list(image_latents.shape[2:])
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
            
    return cached_data

def save_dataset( 
    cached_data: List[Dict], 
    output_path: str,
    dataset_name: str = "cached_embeddings",
    sample_prompt: str = None,
    negative_prompt: str = "blurry, cartoon"
) -> None:

    if not cached_data:
        raise ValueError("No data to save")
        
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_dict = {
        'id': [],
        'image_path': [],
        'caption': [],
        'text_embeds': [],
        'attention_mask': [],
        'image_latents': [],
        'image_size': [],
        'bucket_size': [],
        'latent_size': []
    }
    
    for item in cached_data:
        for key in dataset_dict.keys():
            dataset_dict[key].append(item[key])
    
    dataset = Dataset.from_dict(dataset_dict)
    
    dataset_path = output_path / dataset_name
    dataset.save_to_disk(str(dataset_path))
    
    print(f"Dataset saved to: {dataset_path}")
    print(f"Total samples: {len(cached_data)}")
    
def main():
    parser = argparse.ArgumentParser(description='Cache embeddings for LoRA training')
    parser.add_argument('--image_folder_path', type=str, default=None,
                        help='Path to dataset folder containing images and captions')
    parser.add_argument('--base_resolution', type=int, default=512,
                        choices=[512, 1024],
                        help='Base resolution for aspect ratio buckets (512 or 1024)')
    parser.add_argument('--vae_model_path', type=str, default='ostris/Flex.1-alpha',
                        help='Path to VAE model')
    parser.add_argument('--t5_model_path', type=str, default='ostris/Flex.1-alpha',
                        help='Path to T5 model')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Sample prompt to cache for inference (optional)')
    parser.add_argument('--negative_prompt', type=str, default="blurry, cartoon",
                        help='Negative prompt to use with sample prompt')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for cached embeddings')
    
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    max_length = 512
    assert args.prompt is not None, "Prompt is required"

    text_encoder = TextEncoder(
        device=device,
        dtype=dtype,
        t5_model_path=args.t5_model_path,
        max_length=max_length
    )

    if not args.image_folder_path:
        
        print(f"Caching prompt: '{args.prompt}'")
        cache_prompt_embedding(
        text_encoder=text_encoder,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output_path=args.output if args.output else args.prompt[:10].replace(" ","_") + ".pkl"
        )

    else:
        if args.base_resolution == 512:
            aspect_ratio_buckets = aspect_ratio_buckets_512
        elif args.base_resolution == 1024:
            aspect_ratio_buckets = aspect_ratio_buckets_1024
        else:
            raise ValueError(f"Unsupported base resolution: {args.base_resolution}. Must be 512 or 1024.")
        
        vae = load_vae(device, dtype)

        args.dataset_folder = args.image_folder_path
    
        print(f"Processing dataset folder: {args.dataset_folder}")
        cached_data = process_dataset_folder(
            vae,
            aspect_ratio_buckets,
            text_encoder,
            args.dataset_folder,
        )
        
        print(f"Saving cached dataset to: {args.dataset_folder}")
        save_dataset(
            cached_data,
            f"{args.dataset_folder}",
            "embeddings",
            args.prompt,
            args.negative_prompt
        )

        if args.prompt:
            prompt_cache_path = f"{args.dataset_folder}/embeddings/sample_prompt.pkl"
            print(f"Caching sample prompt: '{args.prompt}'")
            cache_prompt_embedding(
                text_encoder=text_encoder,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                output_path=prompt_cache_path
            )

        print("Caching completed successfully!")

if __name__ == "__main__":
    main() 