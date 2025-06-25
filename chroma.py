import torch
from einops import rearrange
from torch import Tensor

# Flash-Attention 2 (optional)
try:
    from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore
    _HAS_FLASH = True
except (ImportError, ModuleNotFoundError):
    _HAS_FLASH = False


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    # mask should have shape [B, H, L, D]
    if _HAS_FLASH and mask is None and q.is_cuda:
        x = flash_attn_func(
            rearrange(q, "B H L D -> B L H D").contiguous(),
            rearrange(k, "B H L D -> B L H D").contiguous(),
            rearrange(v, "B H L D -> B L H D").contiguous(),
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
        )
        x = rearrange(x, "B L H D -> B H L D")
    else:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device="cpu") / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.cpu(), omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float().to(pos.device)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
import torch.nn.functional as F

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, use_compiled: bool = False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.use_compiled = use_compiled

    def _forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

    def forward(self, x: Tensor):
        return F.rms_norm(x, self.scale.shape, weight=self.scale, eps=1e-6)
        # if self.use_compiled:
        #     return torch.compile(self._forward)(x)
        # else:
        #     return self._forward(x)


def distribute_modulations(tensor: torch.Tensor):
    """
    Distributes slices of the tensor into the block_dict as ModulationOut objects.

    Args:
        tensor (torch.Tensor): Input tensor with shape [batch_size, vectors, dim].
    """
    batch_size, vectors, dim = tensor.shape

    block_dict = {}

    # HARD CODED VALUES! lookup table for the generated vectors
    # TODO: move this into chroma config!
    # Add 38 single mod blocks
    for i in range(38):
        key = f"single_blocks.{i}.modulation.lin"
        block_dict[key] = None

    # Add 19 image double blocks
    for i in range(19):
        key = f"double_blocks.{i}.img_mod.lin"
        block_dict[key] = None

    # Add 19 text double blocks
    for i in range(19):
        key = f"double_blocks.{i}.txt_mod.lin"
        block_dict[key] = None

    # Add the final layer
    block_dict["final_layer.adaLN_modulation.1"] = None
    # 6.2b version
    block_dict["lite_double_blocks.4.img_mod.lin"] = None
    block_dict["lite_double_blocks.4.txt_mod.lin"] = None

    idx = 0  # Index to keep track of the vector slices

    for key in block_dict.keys():
        if "single_blocks" in key:
            # Single block: 1 ModulationOut
            block_dict[key] = ModulationOut(
                shift=tensor[:, idx : idx + 1, :],
                scale=tensor[:, idx + 1 : idx + 2, :],
                gate=tensor[:, idx + 2 : idx + 3, :],
            )
            idx += 3  # Advance by 3 vectors

        elif "img_mod" in key:
            # Double block: List of 2 ModulationOut
            double_block = []
            for _ in range(2):  # Create 2 ModulationOut objects
                double_block.append(
                    ModulationOut(
                        shift=tensor[:, idx : idx + 1, :],
                        scale=tensor[:, idx + 1 : idx + 2, :],
                        gate=tensor[:, idx + 2 : idx + 3, :],
                    )
                )
                idx += 3  # Advance by 3 vectors per ModulationOut
            block_dict[key] = double_block

        elif "txt_mod" in key:
            # Double block: List of 2 ModulationOut
            double_block = []
            for _ in range(2):  # Create 2 ModulationOut objects
                double_block.append(
                    ModulationOut(
                        shift=tensor[:, idx : idx + 1, :],
                        scale=tensor[:, idx + 1 : idx + 2, :],
                        gate=tensor[:, idx + 2 : idx + 3, :],
                    )
                )
                idx += 3  # Advance by 3 vectors per ModulationOut
            block_dict[key] = double_block

        elif "final_layer" in key:
            # Final layer: 1 ModulationOut
            block_dict[key] = [
                tensor[:, idx : idx + 1, :],
                tensor[:, idx + 1 : idx + 2, :],
            ]
            idx += 2  # Advance by 3 vectors

    return block_dict


class Approximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers=4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList(
            [MLPEmbedder(hidden_dim, hidden_dim) for x in range(n_layers)]
        )
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for x in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)

        for layer, norms in zip(self.layers, self.norms):
            x = x + layer(norms(x))

        x = self.out_proj(x)

        return x


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int, use_compiled: bool = False):
        super().__init__()
        self.query_norm = RMSNorm(dim, use_compiled=use_compiled)
        self.key_norm = RMSNorm(dim, use_compiled=use_compiled)
        self.use_compiled = use_compiled

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        use_compiled: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim, use_compiled=use_compiled)
        self.proj = nn.Linear(dim, dim)
        self.use_compiled = use_compiled

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


def _modulation_shift_scale_fn(x, scale, shift):
    return (1 + scale) * x + shift


def _modulation_gate_fn(x, gate, gate_params):
    return x + gate * gate_params


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        use_compiled: bool = False,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_compiled=use_compiled,
        )

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_compiled=use_compiled,
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.use_compiled = use_compiled

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def modulation_shift_scale_fn(self, x, scale, shift):
        if self.use_compiled:
            return torch.compile(_modulation_shift_scale_fn)(x, scale, shift)
        else:
            return _modulation_shift_scale_fn(x, scale, shift)

    def modulation_gate_fn(self, x, gate, gate_params):
        if self.use_compiled:
            return torch.compile(_modulation_gate_fn)(x, gate, gate_params)
        else:
            return _modulation_gate_fn(x, gate, gate_params)

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        distill_vec: list[ModulationOut],
        mask: Tensor,
        txt_weight_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = distill_vec

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        # replaced with compiled fn
        # img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_modulated = self.modulation_shift_scale_fn(
            img_modulated, img_mod1.scale, img_mod1.shift
        )
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        # replaced with compiled fn
        # txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_modulated = self.modulation_shift_scale_fn(
            txt_modulated, txt_mod1.scale, txt_mod1.shift
        )
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe, mask=mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        # replaced with compiled fn
        # img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        # img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        img = self.modulation_gate_fn(img, img_mod1.gate, self.img_attn.proj(img_attn))
        img = self.modulation_gate_fn(
            img,
            img_mod2.gate,
            self.img_mlp(
                self.modulation_shift_scale_fn(
                    self.img_norm2(img), img_mod2.scale, img_mod2.shift
                )
            ),
        )

        # calculate the txt bloks
        # replaced with compiled fn
        # txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        # txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        txt = self.modulation_gate_fn(txt, txt_mod1.gate, self.txt_attn.proj(txt_attn))
        txt = self.modulation_gate_fn(
            txt,
            txt_mod2.gate,
            self.txt_mlp(
                self.modulation_shift_scale_fn(
                    self.txt_norm2(txt), txt_mod2.scale, txt_mod2.shift
                )
            ),
        )

        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        use_compiled: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim, use_compiled=use_compiled)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.use_compiled = use_compiled

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def modulation_shift_scale_fn(self, x, scale, shift):
        if self.use_compiled:
            return torch.compile(_modulation_shift_scale_fn)(x, scale, shift)
        else:
            return _modulation_shift_scale_fn(x, scale, shift)

    def modulation_gate_fn(self, x, gate, gate_params):
        if self.use_compiled:
            return torch.compile(_modulation_gate_fn)(x, gate, gate_params)
        else:
            return _modulation_gate_fn(x, gate, gate_params)

    def forward(
        self, x: Tensor, pe: Tensor, distill_vec: list[ModulationOut], mask: Tensor
    ) -> Tensor:
        mod = distill_vec
        # replaced with compiled fn
        # x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        x_mod = self.modulation_shift_scale_fn(self.pre_norm(x), mod.scale, mod.shift)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, mask=mask)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        # replaced with compiled fn
        # return x + mod.gate * output
        return self.modulation_gate_fn(x, mod.gate, output)


class LastLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_compiled: bool = False,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.use_compiled = use_compiled

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def modulation_shift_scale_fn(self, x, scale, shift):
        if self.use_compiled:
            return torch.compile(_modulation_shift_scale_fn)(x, scale, shift)
        else:
            return _modulation_shift_scale_fn(x, scale, shift)

    def forward(self, x: Tensor, distill_vec: list[Tensor]) -> Tensor:
        shift, scale = distill_vec
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        # replaced with compiled fn
        # x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.modulation_shift_scale_fn(
            self.norm_final(x), scale[:, None, :], shift[:, None, :]
        )
        x = self.linear(x)
        return x

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.utils.checkpoint as ckpt

@dataclass
class ChromaParams:
    in_channels: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    approximator_in_dim: int
    approximator_depth: int
    approximator_hidden_size: int
    _use_compiled: bool


chroma_params = ChromaParams(
    in_channels=64,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    guidance_embed=True,
    approximator_in_dim=64,
    approximator_depth=5,
    approximator_hidden_size=5120,
    _use_compiled=False,
)

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN in {name}")
        quit()


def modify_mask_to_attend_padding(mask, max_seq_length, num_extra_padding=8):
    """
    Modifies attention mask to allow attention to a few extra padding tokens.

    Args:
        mask: Original attention mask (1 for tokens to attend to, 0 for masked tokens)
        max_seq_length: Maximum sequence length of the model
        num_extra_padding: Number of padding tokens to unmask

    Returns:
        Modified mask
    """
    # Get the actual sequence length from the mask
    seq_length = mask.sum(dim=-1)
    batch_size = mask.shape[0]

    modified_mask = mask.clone()

    for i in range(batch_size):
        current_seq_len = int(seq_length[i].item())

        # Only add extra padding tokens if there's room
        if current_seq_len < max_seq_length:
            # Calculate how many padding tokens we can unmask
            available_padding = max_seq_length - current_seq_len
            tokens_to_unmask = min(num_extra_padding, available_padding)

            # Unmask the specified number of padding tokens right after the sequence
            modified_mask[i, current_seq_len : current_seq_len + tokens_to_unmask] = 1

    return modified_mask


class Chroma(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: ChromaParams):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        self.gradient_checkpointing = True
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)

        # TODO: need proper mapping for this approximator output!
        # currently the mapping is hardcoded in distribute_modulations function
        self.distilled_guidance_layer = Approximator(
            params.approximator_in_dim,
            self.hidden_size,
            params.approximator_hidden_size,
            params.approximator_depth,
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    use_compiled=params._use_compiled,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    use_compiled=params._use_compiled,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(
            self.hidden_size,
            1,
            self.out_channels,
            use_compiled=params._use_compiled,
        )

        # TODO: move this hardcoded value to config
        self.mod_index_length = 344
        # self.mod_index = torch.tensor(list(range(self.mod_index_length)), device=0)
        self.register_buffer(
            "mod_index",
            torch.tensor(list(range(self.mod_index_length)), device="cpu"),
            persistent=False,
        )
    
    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def enable_gradient_checkpointing(self, enable: bool = True):
        self.gradient_checkpointing = enable

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        guidance: Tensor,
        attn_padding: int = 1,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        
        # running on sequences img
        img = self.img_in(img)
        txt = self.txt_in(txt)

        # TODO:
        # need to fix grad accumulation issue here for now it's in no grad mode
        # besides, i don't want to wash out the PFP that's trained on this model weights anyway
        # the fan out operation here is deleting the backward graph
        # alternatively doing forward pass for every block manually is doable but slow
        # custom backward probably be better
        with torch.no_grad():
            distill_timestep = timestep_embedding(timesteps, 16)
            # TODO: need to add toggle to omit this from schnell but that's not a priority
            distil_guidance = timestep_embedding(guidance, 16)

            # get all modulation index
            modulation_index = timestep_embedding(self.mod_index, 32)
            # we need to broadcast the modulation index here so each batch has all of the index
            modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1).bfloat16()

            # and we need to broadcast timestep and guidance along too
            timestep_guidance = (
                torch.cat([distill_timestep, distil_guidance], dim=1)
                .unsqueeze(1)
                .repeat(1, self.mod_index_length, 1)
            )

            # then and only then we could concatenate it together
            input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)
            mod_vectors = self.distilled_guidance_layer(input_vec.requires_grad_(True))
        mod_vectors_dict = distribute_modulations(mod_vectors)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # compute mask
        # assume max seq length from the batched input

        max_len = txt.shape[1]

        # mask
        with torch.no_grad():
            txt_mask_w_padding = modify_mask_to_attend_padding(
                txt_mask, max_len, attn_padding
            )
            txt_img_mask = torch.cat(
                [
                    txt_mask_w_padding,
                    torch.ones([img.shape[0], img.shape[1]], device=txt_mask.device),
                ],
                dim=1,
            )
            txt_img_mask = txt_img_mask.float().T @ txt_img_mask.float()
            txt_img_mask = (
                txt_img_mask[None, None, ...]
                .repeat(txt.shape[0], self.num_heads, 1, 1)
                .int()
                .bool()
            )
            # txt_mask_w_padding[txt_mask_w_padding==False] = True

        for i, block in enumerate(self.double_blocks):
            # the guidance replaced by FFN output
            img_mod = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
            txt_mod = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]
            double_mod = [img_mod, txt_mod]

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                img.requires_grad_(True)
                img, txt = ckpt.checkpoint(
                    block, img, txt, pe, double_mod, txt_img_mask
                )
            else:
                img, txt = block(
                    img=img, txt=txt, pe=pe, distill_vec=double_mod, mask=txt_img_mask
                )

        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                img.requires_grad_(True)
                img = ckpt.checkpoint(block, img, pe, single_mod, txt_img_mask)
            else:
                img = block(img, pe=pe, distill_vec=single_mod, mask=txt_img_mask)
        img = img[:, txt.shape[1] :, ...]
        final_mod = mod_vectors_dict["final_layer.adaLN_modulation.1"]
        img = self.final_layer(
            img, distill_vec=final_mod
        )  # (N, T, patch_size ** 2 * out_channels)
        return img

import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    shift_constant = image_seq_len * m + b
    return shift_constant


def time_shift(shift_constant: float, timesteps: torch.Tensor, sigma: float = 1):
    return math.exp(shift_constant) / (
        math.exp(shift_constant) + (1 / timesteps - 1) ** sigma
    )


def vae_flatten(latents):
    # nchw to nhwc then pixel shuffle 2 then flatten
    # n c h w -> n h w c
    # n (h dh) (w dw) c -> n h w (c dh dw)
    # n h w c -> n (h w) c
    # n, c, h, w = latents.shape
    return (
        rearrange(latents, "n c (h dh) (w dw) -> n (h w) (c dh dw)", dh=2, dw=2),
        latents.shape,
    )


def vae_unflatten(latents, shape):
    # reverse of that operator above
    n, c, h, w = shape
    return rearrange(
        latents,
        "n (h w) (c dh dw) -> n c (h dh) (w dw)",
        dh=2,
        dw=2,
        c=c,
        h=h // 2,
        w=w // 2,
    )


def prepare_latent_image_ids(batch_size, height, width):
    # pos embedding for rope, 2d pos embedding, corner embedding and not center based
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    )

    (
        latent_image_id_height,
        latent_image_id_width,
        latent_image_id_channels,
    ) = latent_image_ids.shape

    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
    latent_image_ids = latent_image_ids.reshape(
        batch_size,
        latent_image_id_height * latent_image_id_width,
        latent_image_id_channels,
    )

    return latent_image_ids

import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Chroma,
    # model input
    img: Tensor,
    img_ids: Tensor,
    # guidance
    txt: Tensor,
    # guidance ID
    txt_ids: Tensor,
    # mask
    txt_mask: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 0.0,
):
    # this is ignored for schnell
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            txt_mask=txt_mask,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        img = img + (t_prev - t_curr) * pred

    return img


def denoise_batched_timesteps(
    model: Chroma,
    # model input
    img: Tensor,
    img_ids: Tensor,
    # guidance
    txt: Tensor,
    # guidance ID
    txt_ids: Tensor,
    # mask
    txt_mask: Tensor,
    # sampling parameters
    timesteps: Tensor,  # Shape: (B, N), where N is the number of time points
    guidance: float = 4.0,
):
    """
    Performs ODE solving using the Euler method with potentially different
    timestep sequences for each sample in the batch.

    Args:
        model: The flow matching model.
        img: Input tensor (e.g., noise) shape (B, C, H, W).
        img_ids: Image IDs tensor, shape (B, ...).
        txt: Text conditioning tensor, shape (B, L, D).
        txt_ids: Text IDs tensor, shape (B, L).
        txt_mask: Text mask tensor, shape (B, L).
        timesteps: Tensor containing the time points for each batch sample.
                   Shape (B, N), where B is the batch size and N is the
                   number of time points (e.g., [t_start, ..., t_end]).
                   Time should generally decrease (e.g., [1.0, 0.8, ..., 0.0]).
        guidance: Classifier-free guidance strength.
    Returns:
        Denoised image tensor, shape (B, C, H, W).
    """
    batch_size = img.shape[0]
    num_time_points = timesteps.shape[1]
    num_steps = num_time_points - 1  # Number of integration steps

    if timesteps.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: img has {batch_size}, "
            f"but timesteps has {timesteps.shape[0]}"
        )
    if timesteps.ndim != 2:
        raise ValueError(
            f"timesteps tensor must be 2D (B, N), but got shape {timesteps.shape}"
        )

    # Guidance vector remains the same for all elements in this specific call
    guidance_vec = torch.full(
        (batch_size,), guidance, device=img.device, dtype=img.dtype
    )

    # Ensure timesteps tensor is on the same device and dtype as img
    timesteps = timesteps.to(device=img.device, dtype=img.dtype)

    # Iterate through the integration steps (from step 0 to N-2)
    for i in range(num_steps):
        # Get the current time for each batch element
        t_curr_batch = timesteps[:, i]  # Shape: (B,)
        # Get the next time for each batch element
        t_next_batch = timesteps[:, i + 1]  # Shape: (B,)

        # Model prediction using the current time for each batch element
        # Your model already accepts batched timesteps (shape B,)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            txt_mask=txt_mask,
            timesteps=t_curr_batch,  # Pass the (B,) tensor of current times
            guidance=guidance_vec,
        )

        # Calculate the step size (dt) for each batch element
        # dt = t_next - t_curr (Note: if time goes 1->0, dt will be negative)
        dt_batch = t_next_batch - t_curr_batch  # Shape: (B,)

        # Reshape dt for broadcasting: (B,) -> (B, 1, 1)
        dt_batch_reshaped = dt_batch.view(batch_size, 1, 1)

        # Euler step update: x_{t+1} = x_t + dt * v(x_t, t)
        img = img + dt_batch_reshaped * pred

    return img


def denoise_cfg(
    model: Chroma,
    # model input
    img: Tensor,
    img_ids: Tensor,
    # guidance
    txt: Tensor,
    neg_txt: Tensor,
    # guidance ID
    txt_ids: Tensor,
    neg_txt_ids: Tensor,
    # mask
    txt_mask: Tensor,
    neg_txt_mask: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    cfg: float = 2.0,
    first_n_steps_without_cfg: int = 4, 
):
    # this is ignored for schnell
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )
    step_count = 0
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            txt_mask=txt_mask,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        # disable cfg for x steps before using cfg
        if step_count < first_n_steps_without_cfg or first_n_steps_without_cfg == -1:
            img = img.to(pred) + (t_prev - t_curr) * pred
        else:
            pred_neg = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                txt_mask=neg_txt_mask,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            pred_cfg = pred_neg + (pred - pred_neg) * cfg

            img = img + (t_prev - t_curr) * pred_cfg

        step_count += 1

    return img


def denoise_cfg_batched_timesteps(
    model: Chroma,
    # model input
    img: Tensor,
    img_ids: Tensor,
    # guidance
    txt: Tensor,
    neg_txt: Tensor,
    # guidance ID
    txt_ids: Tensor,
    neg_txt_ids: Tensor,
    # mask
    txt_mask: Tensor,
    neg_txt_mask: Tensor,
    # sampling parameters
    timesteps: Tensor,  # Shape: (B, N), where N is the number of time points
    guidance: float = 0.0,
    cfg: float = 2.0,
    first_n_steps_without_cfg: int = 4,
):
    """
    Performs ODE solving using the Euler method with Classifier-Free Guidance (CFG)
    and potentially different timestep sequences for each sample in the batch.

    Args:
        model: The flow matching model.
        img: Input tensor (e.g., noise) shape (B, C, H, W).
        img_ids: Image IDs tensor, shape (B, ...).
        txt: Positive text conditioning tensor, shape (B, L, D).
        neg_txt: Negative text conditioning tensor, shape (B, L, D).
        txt_ids: Positive text IDs tensor, shape (B, L).
        neg_txt_ids: Negative text IDs tensor, shape (B, L).
        txt_mask: Positive text mask tensor, shape (B, L).
        neg_txt_mask: Negative text mask tensor, shape (B, L).
        timesteps: Tensor containing the time points for each batch sample.
                   Shape (B, N), where B is the batch size and N is the
                   number of time points (e.g., [t_start, ..., t_end]).
                   Time should generally decrease (e.g., [1.0, 0.8, ..., 0.0]).
        guidance: Guidance strength passed to the model (potentially ignored).
        cfg: Classifier-Free Guidance scale. A value of 1.0 disables CFG.
        first_n_steps_without_cfg: The number of initial integration steps
                                   (intervals) for which CFG will *not* be
                                   applied, even if cfg > 1.0. Set to 0 to
                                   apply CFG from the start, or -1 to always
                                   apply CFG (if cfg > 1.0).
    Returns:
        Denoised image tensor, shape (B, C, H, W).
    """
    batch_size = img.shape[0]
    num_time_points = timesteps.shape[1]
    num_steps = num_time_points - 1  # Number of integration steps

    # --- Input Validation ---
    if timesteps.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: img has {batch_size}, "
            f"but timesteps has {timesteps.shape[0]}"
        )
    if timesteps.ndim != 2:
        raise ValueError(
            f"timesteps tensor must be 2D (B, N), but got shape {timesteps.shape}"
        )
    # Check consistency of conditioning tensors
    for name, tensor in [
        ("txt", txt),
        ("neg_txt", neg_txt),
        ("txt_ids", txt_ids),
        ("neg_txt_ids", neg_txt_ids),
        ("txt_mask", txt_mask),
        ("neg_txt_mask", neg_txt_mask),
    ]:
        if tensor.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: img has {batch_size}, "
                f"but {name} has {tensor.shape[0]}"
            )
    # --- End Validation ---

    # Guidance vector (its effect depends on the model)
    guidance_vec = torch.full(
        (batch_size,), guidance, device=img.device, dtype=img.dtype
    )

    # Ensure timesteps tensor is on the same device and dtype as img
    timesteps = timesteps.to(device=img.device, dtype=img.dtype)

    # Iterate through the integration steps (intervals)
    for i in range(num_steps):
        # Get the current time for each batch element
        t_curr_batch = timesteps[:, i]  # Shape: (B,)
        # Get the next time for each batch element
        t_next_batch = timesteps[:, i + 1]  # Shape: (B,)

        # --- Positive Prediction ---
        pred_pos = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            txt_mask=txt_mask,
            timesteps=t_curr_batch,  # Batched timesteps
            guidance=guidance_vec,
        )

        # --- CFG Logic ---
        # Determine if CFG should be applied in this step
        # Apply CFG if cfg > 1.0 AND (we are past the initial steps OR first_n_steps_without_cfg is -1)
        apply_cfg = cfg > 1.0 and (
            i >= first_n_steps_without_cfg or first_n_steps_without_cfg == -1
        )

        if apply_cfg:
            # --- Negative Prediction ---
            pred_neg = model(
                img=img,  # Use the *same* input image state as for positive pred
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                txt_mask=neg_txt_mask,
                timesteps=t_curr_batch,  # Use the same batched timesteps
                guidance=guidance_vec,  # Pass guidance here too
            )
            # Combine predictions using CFG formula
            # pred = uncond + cfg * (cond - uncond)
            pred_final = pred_neg + cfg * (pred_pos - pred_neg)
        else:
            # If not applying CFG, use the positive prediction directly
            pred_final = pred_pos
        # --- End CFG Logic ---

        # Calculate the step size (dt) for each batch element
        dt_batch = t_next_batch - t_curr_batch  # Shape: (B,)

        # Reshape dt for broadcasting: (B,) -> (B, 1, 1)
        dt_batch_reshaped = dt_batch.view(batch_size, 1, 1)

        # Euler step update: x_{t+1} = x_t + dt * v(x_t, t)
        # Ensure img is on the correct device/dtype if pred_final changes it (unlikely but safe)
        img = img.to(pred_final) + dt_batch_reshaped * pred_final

    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

def create_distribution(num_points, device=None):
    x = torch.linspace(0, 1, num_points, device=device)

    probabilities = -7.7 * ((x - 0.5) ** 2) + 2

    probabilities /= probabilities.sum()

    return x, probabilities

def sample_from_distribution(x, probabilities, num_samples, device=None):
    """Sample timesteps from custom distribution using inverse CDF sampling"""
    cdf = torch.cumsum(probabilities, dim=0)

    uniform_samples = torch.rand(num_samples, device=device)

    indices = torch.searchsorted(cdf, uniform_samples, right=True)

    sampled_values = x[indices]

    return sampled_values
