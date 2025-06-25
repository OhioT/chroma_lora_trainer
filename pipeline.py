from typing import Union, List, Optional
import torch
from PIL import Image
from chroma import Chroma, chroma_params, prepare_latent_image_ids, vae_flatten, get_noise, get_schedule, denoise_cfg, unpack

from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from diffusers import AutoencoderKL
from transformers import T5TokenizerFast, T5EncoderModel

def sample_image(
        transformer,
        vae,
        cached_embeddings,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        print(f"  Positive prompt: '{cached_embeddings['positive']['prompt_text']}'")
        print(f"  Negative prompt: '{cached_embeddings['negative']['prompt_text']}'")
            
        prompt_embeds = cached_embeddings['positive']['text_embeds'].to(transformer.device, dtype=transformer.dtype)
        prompt_attn_mask = cached_embeddings['positive']['attention_mask'].to(transformer.device)
        negative_prompt_embeds = cached_embeddings['negative']['text_embeds'].to(transformer.device, dtype=transformer.dtype)
        negative_prompt_attn_mask = cached_embeddings['negative']['attention_mask'].to(transformer.device)
        
        batch_size = prompt_embeds.shape[0]

        device = transformer.device
        vae_scale_factor = 8
        
        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3, device=device, dtype=torch.bfloat16)
        negative_text_ids = torch.zeros(batch_size, negative_prompt_embeds.shape[1], 3, device=device, dtype=torch.bfloat16)

        latents = get_noise(
            num_samples=batch_size,
            height=height,
            width=width,
            device=device,
            dtype=prompt_embeds.dtype,
            seed=generator.initial_seed() if generator else 42
        )

        latents, _ = vae_flatten(latents)
        
        h = height // vae_scale_factor
        w = width // vae_scale_factor
        latent_image_ids = prepare_latent_image_ids(batch_size, h, w).to(device)

        h = height // vae_scale_factor
        w = width // vae_scale_factor
        image_seq_len = (h // 2) * (w // 2)
        
        timesteps = get_schedule(
            num_steps=num_inference_steps,
            image_seq_len=image_seq_len,
            base_shift=0.5,
            max_shift=1.15,
            shift=True
        )

        print(f"Running {num_inference_steps} inference steps...")
        
        latents = denoise_cfg(
            model=transformer,
            img=latents,
            img_ids=latent_image_ids,
            txt=prompt_embeds,
            neg_txt=negative_prompt_embeds,
            txt_ids=text_ids,
            neg_txt_ids=negative_text_ids,
            txt_mask=prompt_attn_mask,
            neg_txt_mask=negative_prompt_attn_mask,
            timesteps=timesteps,
            guidance=0.0, 
            cfg=guidance_scale,
            first_n_steps_without_cfg=4
        )

        latents = unpack(latents, height, width)
        
        with torch.no_grad():
            latents = latents / vae.config.scaling_factor
            image = vae.decode(latents, return_dict=False)[0]
        
        image = (image / 2 + 0.5).clamp(0, 1)
        
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(img) for img in image]

        return image[0]

def apply_lora_to_model(transformer, lora_rank, lora_alpha, device='cuda', dtype=torch.bfloat16):

    target_modules = [
        "qkv", "proj","img_attn.qkv", "img_attn.proj","txt_attn.qkv", "txt_attn.proj","img_mlp.0", "img_mlp.2","txt_mlp.0", "txt_mlp.2","linear1", "linear2","linear"
    ]
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,  
        target_modules=target_modules,
        lora_dropout=0.0, 
        bias="none",
    )
    
    transformer = get_peft_model(transformer, lora_config)
    
    print(f"LoRA applied successfully with rank {lora_rank}")
    return transformer

def load_chroma_model(model_path, device='cuda', dtype=torch.bfloat16, lora_rank=None, lora_alpha=None):

    print("Loading Chroma transformer...")

    dtype = torch.bfloat16

    state_dict = load_file(model_path, device='cpu')

    original_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    with torch.device('cuda'):
        transformer = Chroma(chroma_params)
    torch.set_default_dtype(original_default_dtype)

    transformer = transformer.to(device=device)

    transformer.load_state_dict(state_dict)
    print("Loaded state dict")

    transformer.to(device)
    
    if lora_rank is not None:
        transformer = apply_lora_to_model(transformer, lora_rank, lora_alpha, device, dtype)
    
    return transformer

def load_vae(device='cuda', dtype=torch.bfloat16):
    vae = AutoencoderKL.from_pretrained(
            "ostris/Flex.1-alpha",
            subfolder="vae",
            torch_dtype=dtype
        )
    vae.to(device, dtype=dtype)
    vae.eval()
    vae.requires_grad_(False) 
    return vae

class TextEncoder:

    def __init__(
        self,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        t5_model_path: str = 'ostris/Flex.1-alpha',
        max_length: int = 512
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_length = max_length
        
        print("Loading T5 text encoder...")
        self.tokenizer = T5TokenizerFast.from_pretrained(
            t5_model_path, 
            subfolder="tokenizer_2", 
            torch_dtype=self.dtype
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            t5_model_path, 
            subfolder="text_encoder_2", 
            torch_dtype=self.dtype
        )
        self.text_encoder.to(self.device, dtype=self.dtype)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
    
    def encode_text(self, prompts):
        """Encode text prompts using T5"""
        if isinstance(prompts, str):
            prompts = [prompts]
            
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            text_embeds = self.text_encoder(
                text_input_ids, 
                output_hidden_states=False,
                attention_mask=attention_mask
            )[0]
            
            text_embeds = text_embeds.to(dtype=self.dtype, device='cpu')
            attention_mask = attention_mask.to(device='cpu')
            text_input_ids = text_input_ids.to(device='cpu')

        return text_embeds, attention_mask, text_input_ids

aspect_ratio_buckets_1024 = [
    (512, 2048),
    (576, 1664),
    (640, 1536),
    (704, 1408),
    (768, 1280),
    (832, 1216),
    (896, 1152),
    (960, 1088),
    (1024, 1024),
    (1088, 960),
    (1152, 896),
    (1216, 832),
    (1280, 768),
    (1408, 704),
    (1536, 640),
    (1664, 576),
    (2048, 512),
]

aspect_ratio_buckets_512 = [
    (256, 1024),
    (288, 832),
    (320, 768),
    (352, 704),
    (384, 640),
    (416, 608),
    (448, 576),
    (480, 544),
    (512, 512),
    (544, 480),
    (576, 448),
    (608, 416),
    (640, 384),
    (704, 352),
    (768, 320),
    (832, 288),
    (1024, 256),
]

    
