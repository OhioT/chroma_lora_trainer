import torch
import argparse
import pickle
import os
from pathlib import Path
from typing import Optional
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange, repeat
from diffusers import AutoencoderKL
from safetensors.torch import save_file

from pipeline import sample_image, load_chroma_model
from chroma import create_distribution, sample_from_distribution

class LoRAChromaTrainer:
    def __init__(
        self,
        args
    ):
        self.model_path = args.model_path
        self.dataset_path = args.dataset_path
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_folder_path = args.image_folder_path
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.name = args.name

        self.save_every_n_epochs = args.save_every_n_epochs
        
        self.lora_rank = args.lora_rank
        self.lora_alpha = args.lora_alpha
        self.lora_dropout = args.lora_dropout
        self.target_modules = ["qkv", "proj", "img_attn.qkv", "img_attn.proj","txt_attn.qkv", "txt_attn.proj","img_mlp.0", "img_mlp.2","txt_mlp.0", "txt_mlp.2","linear1", "linear2","linear"]
        
        self.sample_every_n_epochs = args.sample_every_n_epochs
        self.cached_embeddings_path = args.cached_embeddings_path
        self.sample_width = args.sample_width
        self.sample_height = args.sample_height
        self.sample_steps = args.sample_steps
        self.sample_guidance_scale = args.sample_guidance_scale
        self.sample_seed = args.sample_seed
        
        self.transformer = None
        self.vae = None
        self.optimizer = None
        self.lr_scheduler = None
        self.cached_embeddings = None
        
        self.sample_output_dir = self.output_dir / "samples"
        self.sample_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.global_step = 0
        self.epoch = 0
        
        self.timestep_x, self.timestep_probabilities = create_distribution(
            1000, device=self.device
        )
        
    def load_model_components(self):
        self.transformer = load_chroma_model(self.model_path, device=self.device, dtype=self.dtype, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha)
        
        self.transformer.print_trainable_parameters()
        
        for name, param in self.transformer.named_parameters():
            if param.requires_grad and "lora_" in name:
                param.data = param.data.to(torch.float32)
        
        print("Loading text encoders...")
        
        extras_path = 'ostris/Flex.1-alpha'
        
        print("Loading VAE...")
        
        self.vae = AutoencoderKL.from_pretrained(
            extras_path,
            subfolder="vae",
            torch_dtype=self.dtype
        )
        self.vae.to(self.device, dtype=self.dtype)
        self.vae.eval()
        self.vae.requires_grad_(False)

        with open(self.cached_embeddings_path, 'rb') as f:
            self.cached_embeddings = pickle.load(f)

    def setup_optimizer(self):
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0,
            eps=1e-8
        )

        total_steps = len(self.train_dataloader) * self.num_epochs
        warmup_steps = int(0.1 * total_steps) 
        
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.learning_rate * 0.1
        )
        
    def setup_dataloader(self):
        print(f"Loading dataset from {self.dataset_path}")
        
        dataset = Dataset.load_from_disk(self.dataset_path).with_format('torch')
        
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"Dataset loaded with {len(dataset)} samples")
        
    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor, 
        text_embeddings,
        attention_mask,
    ):
        with torch.no_grad():
            bs, c, h, w = latent_model_input.shape
            latent_model_input_packed = rearrange(
                latent_model_input,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=2,
                pw=2
            )

            img_ids = torch.zeros(h // 2, w // 2, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
            img_ids = repeat(img_ids, "h w c -> b (h w) c",
                             b=bs).to(self.device)

            txt_ids = torch.zeros(
                bs, text_embeddings.shape[1], 3).to(self.device)

            guidance = torch.full([1], 0, device=self.device, dtype=torch.float32)
            guidance = guidance.expand(latent_model_input_packed.shape[0])

            cast_dtype = latent_model_input_packed.dtype

        noise_pred = self.transformer(
            img=latent_model_input_packed.to(
                self.device, cast_dtype
            ),
            img_ids=img_ids.to(self.device, cast_dtype),
            txt=text_embeddings.to(
                self.device, cast_dtype
            ),
            txt_ids=txt_ids.to(self.device, cast_dtype),
            txt_mask=attention_mask.to(
                self.device, cast_dtype
            ),
            timesteps=timestep.to(self.device, cast_dtype) / 1000,
            guidance=guidance.to(self.device, cast_dtype)
        )

        #if isinstance(noise_pred, QTensor):
        #    noise_pred = noise_pred.dequantize()

        noise_pred = rearrange(
            noise_pred,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=latent_model_input.shape[2] // 2,
            w=latent_model_input.shape[3] // 2,
            ph=2,
            pw=2,
            c=self.vae.config.latent_channels
        )
        
        return noise_pred
    
    def train_step(self, batch):
        """Single training step"""

        latents = batch['image_latents'].to(self.device, dtype=self.dtype).flatten(0,1)
        text_embeds = batch['text_embeds'].to(self.device, dtype=self.dtype).flatten(0,1)
        attention_mask = batch['attention_mask'].to(self.device, dtype=self.dtype).flatten(0,1)

        timesteps = sample_from_distribution(
            self.timestep_x, self.timestep_probabilities, 
            latents.shape[0], device=self.device
        )

        timesteps = (timesteps * 1000).long()
        
        noise = torch.randn_like(latents)
        
        t = timesteps.to(noise.dtype) / 1000.0
        t = t.view(-1, 1, 1, 1)
        
        noisy_latents = (1.0 - t) * latents + t * noise
        
        target = noise - latents
        
        noise_pred = self.get_noise_prediction(noisy_latents, timesteps, text_embeds, attention_mask)
            
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none").mean()
        
        return loss
    
    def save_checkpoint(self, step: Optional[int] = None):
        if step is None:
            step = self.global_step
            
        lora_state_dict = self.transformer.state_dict()
        
        lora_state_dict = {k.replace("base_model.model", "diffusion_model").replace("default.", ""): v for k, v in lora_state_dict.items() if "lora_" in k}
        
        checkpoint_path = self.output_dir / f"lora_{self.name}_step_{step}.safetensors"
        save_file(lora_state_dict, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        
    def generate_sample(self, epoch: int):

        print(f"Generating sample for epoch {epoch}...")
        
        self.transformer.eval()
        
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.sample_seed)

        with torch.no_grad():
            image = sample_image(
                transformer=self.transformer,
                vae=self.vae,
                cached_embeddings=self.cached_embeddings,
                height=self.sample_height,
                width=self.sample_width,
                num_inference_steps=self.sample_steps,
                guidance_scale=self.sample_guidance_scale,
                generator=generator
            )
            
            sample_path = self.sample_output_dir / f"epoch_{epoch:04d}_step_{self.global_step:06d}.png"
            image.save(sample_path)

        self.transformer.train()

    def train(self):
        print("Starting LoRA training...")

        self.load_model_components()
        self.setup_dataloader()
        self.setup_optimizer()

        self.transformer.train()
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0

            if epoch % self.sample_every_n_epochs == 0:
                self.generate_sample(epoch + 1)
            
            if epoch % self.save_every_n_epochs == 0:
                self.save_checkpoint()
            
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1}/{self.num_epochs}"
            )
            
            self.optimizer.zero_grad()
            
            for _, batch in enumerate(progress_bar):

                loss = self.train_step(batch)
                
                loss.backward()
                
                epoch_loss += loss.item()
                
                trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
                
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                self.optimizer.step()
                
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.6f}",
                    'lr': f"{current_lr:.2e}",
                    'step': self.global_step
                })
            
            avg_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}")

        self.save_checkpoint()
        
        print("Training completed!")

def main():
    parser = argparse.ArgumentParser(description='Train LoRA on Chroma model using cached embeddings')
    
    parser.add_argument('--name', type=str, required=True,
                        help='Name of the LoRA')
    parser.add_argument('--image_folder_path', type=str, required=True,
                        help='Path to image folder with cached embeddings dataset')
    parser.add_argument('--model_path', type=str, default='chroma-unlocked-v39.safetensors',
                        help='Path to Chroma model safetensors file')
    parser.add_argument('--cached_embeddings_path', type=str, default=None,
                        help='Path to cached prompt embeddings for sampling (.pkl file)')
    
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Number of training epochs')

    parser.add_argument('--save_every_n_epochs', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--sample_every_n_epochs', type=int, default=10,
                        help='Generate sample every N epochs (0 to disable)')

    parser.add_argument('--sample_steps', type=int, default=25,
                        help='Number of inference steps for sampling')
    parser.add_argument('--sample_guidance_scale', type=float, default=5.0,
                        help='Guidance scale for sampling')
    parser.add_argument('--sample_seed', type=int, default=42,
                        help='Random seed for sampling')
    parser.add_argument('--sample_width', type=int, default=512,
                        help='Sample image width')
    parser.add_argument('--sample_height', type=int, default=512,
                        help='Sample image height')
    
    args = parser.parse_args()
    
    args.dataset_path = f'{args.image_folder_path}/embeddings'
    args.output_dir = f'lora/{args.name}'

    if not args.cached_embeddings_path:
        args.cached_embeddings_path = f"{args.image_folder_path}/embeddings/sample_prompt.pkl"

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path {args.dataset_path} does not exist")
    if not os.path.exists(args.cached_embeddings_path):
        raise FileNotFoundError(f"Cached embeddings path {args.cached_embeddings_path} does not exist. You may need to cache the embeddings first.")
    
    trainer = LoRAChromaTrainer(args)
    
    trainer.train()

if __name__ == "__main__":
    main() 