import torch
import pickle
import argparse
from pipeline import load_vae, sample_image, load_chroma_model
from safetensors.torch import load_file

def main():
    parser = argparse.ArgumentParser(description='Sample from Chroma model')
    parser.add_argument('--model_path', type=str, default="chroma-unlocked-v39.safetensors",
                        help='Path to Chroma model safetensors file')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Path to LoRA weights safetensors file (optional)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Path to cached prompt embeddings')
    parser.add_argument('--output', type=str, default="generated_image.png",
                        help='Output image path')
    parser.add_argument('--width', type=int, default=512,
                        help='Image width')
    parser.add_argument('--height', type=int, default=512,
                        help='Image height')
    parser.add_argument('--steps', type=int, default=25,
                        help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=4.0,
                        help='Guidance scale')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()

    device = 'cuda'
    dtype = torch.bfloat16
    
    if args.lora_path:
        lora_state_dict = load_file(args.lora_path, device='cpu')
        lora_rank = next(iter(lora_state_dict.items()))[1].shape[0]
        lora = {k.replace("diffusion_model", "base_model.model").replace(".weight", ".default.weight"): v for k, v in lora_state_dict.items() if "lora_" in k}
        transformer = load_chroma_model(args.model_path, device, dtype, lora_rank=lora_rank, lora_alpha=lora_rank)
        transformer.load_state_dict(lora, strict=False)
    else:
        transformer = load_chroma_model(args.model_path, device, dtype)
    
    vae = load_vae(device, dtype)
    
    assert args.prompt is not None, "Cached embeddings are required"
    with open(args.prompt, 'rb') as f:
        cached_embeddings = pickle.load(f)
    
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    with torch.no_grad():
        image = sample_image(
            transformer=transformer,
            vae=vae,
            cached_embeddings=cached_embeddings,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator
        )
    
    image.save(args.output)
    print(f"Image saved to: {args.output}")
        
if __name__ == "__main__":
    main() 