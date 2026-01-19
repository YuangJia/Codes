import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from einops import rearrange
import math
from models.modules.dit_modules.math import attention, rope

# 假设这些基础模块都在 models.modules.dit_modules.layers 中
# 为了代码完整性，我保留了 Modulation, LastLayer 等关键定义，并补充了新的 TextCrossAttnBlock

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
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t.dtype)
    return embedding


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
    

class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor

class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class SingleStreamBlockWithText(nn.Module):
    """
    Flux-style Single Stream Block with extra Text Cross-Attention.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qk_scale: float | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # --- Parallel Attention + MLP ---
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        
        self.norm = QKNorm(head_dim) # For Self-Attention QK
        
        # Pre-Norm (LayerNorm without affine, controlled by AdaLN)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

        # --- Text Cross-Attention ---
        self.text_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.text_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        # Simple AdaLN for Text Norm (Scale & Shift only)
        self.text_mod_lin = nn.Linear(hidden_size, 2 * hidden_size, bias=True)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, text_tokens: Tensor) -> Tensor:
        # 1. Base Modulation
        mod, _ = self.modulation(vec)
        
        # 2. Pre-Norm & Modulate
        x_mod = self.pre_norm(x)
        x_mod = modulate(x_mod, mod.shift, mod.scale)
        
        # 3. QKV & MLP Split
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        
        # 4. Self-Attention
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        
        # Note: attention function handles rearrange back
        x_attn = attention(q, k, v, pe=pe) 
        
        # 5. Text Cross-Attention
        # Calculate modulation for text norm
        text_scale, text_shift = self.text_mod_lin(F.silu(vec))[:, None, :].chunk(2, dim=-1)
        
        x_for_text = self.text_norm(x)
        x_for_text = modulate(x_for_text, text_shift, text_scale)
        
        # Standard Cross-Attention
        x_text_attn = self.text_attn(x_for_text, text_tokens, text_tokens)[0]
        
        # 6. Fuse (Add text attention to self attention output)
        x_attn_combined = x_attn + x_text_attn
        
        # 7. Final Projection
        output = self.linear2(torch.cat((x_attn_combined, self.mlp_act(mlp)), 2))
        
        return x + mod.gate * output

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        # Note: attention function handles rearrange back
        x = attention(q, k, v, pe=pe) 
        x = self.proj(x)
        return x


# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# -----------------------------------------------------------------------------
# DoubleStreamBlock (Traj & Visual 联合注意力)
# -----------------------------------------------------------------------------    
class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # --- Stream 1: Image/Trajectory ---
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias) # Helper for QKV projection

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # --- Stream 2: Condition/Visual ---
        self.cond_mod = Modulation(hidden_size, double=True)
        self.cond_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cond_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.cond_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, cond: Tensor, vec: Tensor, pe: Tensor = None) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        cond_mod1, cond_mod2 = self.cond_mod(vec)

        # 1. Prepare Image QKV
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, img_mod1.shift, img_mod1.scale)
        # Manually using the sub-modules of SelfAttention class to get QKV
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # 2. Prepare Cond QKV
        cond_modulated = self.cond_norm1(cond)
        cond_modulated = modulate(cond_modulated, cond_mod1.shift, cond_mod1.scale)
        cond_qkv = self.cond_attn.qkv(cond_modulated)
        cond_q, cond_k, cond_v = rearrange(cond_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        cond_q, cond_k = self.cond_attn.norm(cond_q, cond_k, cond_v)

        # 3. Joint Attention
        q = torch.cat((cond_q, img_q), dim=2)
        k = torch.cat((cond_k, img_k), dim=2)
        v = torch.cat((cond_v, img_v), dim=2)

        # Run attention
        attn = attention(q, k, v, pe=pe) # Returns (B, L_total, D)
        
        # Split back
        cond_len = cond.shape[1]
        cond_attn, img_attn = attn[:, :cond_len], attn[:, cond_len:]

        # 4. Update Image Stream (Residual + MLP)
        # Attention projection
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        # MLP
        img = img + img_mod2.gate * self.img_mlp(
            modulate(self.img_norm2(img), img_mod2.shift, img_mod2.scale)
        )

        # 5. Update Cond Stream
        cond = cond + cond_mod1.gate * self.cond_attn.proj(cond_attn)
        cond = cond + cond_mod2.gate * self.cond_mlp(
            modulate(self.cond_norm2(cond), cond_mod2.shift, cond_mod2.scale)
        )

        return img, cond

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

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output