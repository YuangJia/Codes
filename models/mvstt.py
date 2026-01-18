import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Tuple, Union, List
from models.modules.dit_modules.embedder import PatchEmbed3D

# ==========================================
# 0. 辅助工具
# ==========================================

# ==========================================
# 0.1 RoPE 相关函数
# ==========================================
def get_1d_rotary_pos_embed(
    dim: int,
    length: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成 1D RoPE embeddings (用于 Time 维度)

    Args:
        dim: Head dimension
        length: Sequence length (时间步数 T)
        theta: Scaling factor
    Returns:
        freqs_cos, freqs_sin: [length, dim] each
    """
    pos = torch.arange(length).float()
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [D/2]
    freqs = torch.outer(pos, freqs)  # [T, D/2]

    freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [T, D]
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [T, D]
    return freqs_cos, freqs_sin


def get_2d_rotary_pos_embed(
    dim: int,
    grid_h: int,
    grid_w: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成 2D RoPE embeddings (用于 Space 维度)

    Args:
        dim: Head dimension (will be split between H and W)
        grid_h: Height of grid
        grid_w: Width of grid
        theta: Scaling factor
    Returns:
        freqs_cos, freqs_sin: [H*W, dim] each
    """
    # Split dim between H and W dimensions
    dim_h = dim // 2
    dim_w = dim - dim_h

    # Generate 1D embeddings for H and W
    pos_h = torch.arange(grid_h).float()
    pos_w = torch.arange(grid_w).float()

    freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
    freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))

    freqs_h = torch.outer(pos_h, freqs_h)  # [H, dim_h/2]
    freqs_w = torch.outer(pos_w, freqs_w)  # [W, dim_w/2]

    # Create 2D grid
    freqs_h = freqs_h.unsqueeze(1).repeat(1, grid_w, 1)  # [H, W, dim_h/2]
    freqs_w = freqs_w.unsqueeze(0).repeat(grid_h, 1, 1)  # [H, W, dim_w/2]

    # Flatten spatial dimensions
    freqs_h = freqs_h.reshape(-1, dim_h // 2)  # [H*W, dim_h/2]
    freqs_w = freqs_w.reshape(-1, dim_w // 2)  # [H*W, dim_w/2]

    # Apply cos/sin and interleave
    freqs_h_cos = freqs_h.cos().repeat_interleave(2, dim=1)  # [H*W, dim_h]
    freqs_h_sin = freqs_h.sin().repeat_interleave(2, dim=1)
    freqs_w_cos = freqs_w.cos().repeat_interleave(2, dim=1)  # [H*W, dim_w]
    freqs_w_sin = freqs_w.sin().repeat_interleave(2, dim=1)

    # Concatenate H and W embeddings
    freqs_cos = torch.cat([freqs_h_cos, freqs_w_cos], dim=1)  # [H*W, dim]
    freqs_sin = torch.cat([freqs_h_sin, freqs_w_sin], dim=1)

    return freqs_cos, freqs_sin


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(-2)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to Q and K.

    Args:
        q: [B, num_heads, Seq, head_dim]
        k: [B, num_heads, Seq, head_dim]
        freqs_cos: [Seq, head_dim]
        freqs_sin: [Seq, head_dim]
    Returns:
        q_rot, k_rot: Same shape as inputs
    """
    # Reshape freqs: [1, 1, Seq, head_dim]
    cos = freqs_cos.unsqueeze(0).unsqueeze(0)
    sin = freqs_sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation
    q_rot = (q.float() * cos + rotate_half(q.float()) * sin).type_as(q)
    k_rot = (k.float() * cos + rotate_half(k.float()) * sin).type_as(k)

    return q_rot, k_rot


# ==========================================
# 0.2 Fourier Embeddings (原有的)
# ==========================================
def get_fourier_embeds_from_coordinates(embed_dim, coordinates, max_period: int = 100):
    """
    Args:
        embed_dim: int
        coordinates: [B x N x C] (Batch, Seq_Len, Dim)
    Returns:
        [B x N x C x embed_dim]
    """
    half_embed_dim = embed_dim // 2
    coordinates = coordinates.float()

    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half_embed_dim, dtype=torch.float32) / half_embed_dim
    ).to(coordinates.device)

    emb = coordinates.unsqueeze(-1) * freqs.view(1, 1, 1, -1)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if embed_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)

    return emb


# ==========================================
# 1. 配置类
# ==========================================
class MVConfig:
    def __init__(self,
                 input_vae_dim=32,
                 embed_dim=1024,
                 n_views=6,
                 n_head=16,
                 n_layer=4,
                 latent_size=(4, 32, 32),  # t, h, w (VAE输出的时间维度)
                 time_compression_rate=4,  # T_raw = t * 4
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 patch_size=(1, 2, 2),
                 # 轨迹配置
                 pose_x_vocab_size=512,
                 pose_y_vocab_size=512,
                 pose_z_vocab_size=512,
                 yaw_vocab_size=512,
                 yaw_pose_emb_dim=512,
                 # RoPE 配置
                 use_rope=True,
                 rope_theta=10000.0):

        self.input_vae_dim = input_vae_dim
        self.embed_dim = embed_dim
        self.n_views = n_views
        self.n_head = n_head
        self.n_layer = n_layer
        self.latent_size = latent_size
        self.time_compression_rate = time_compression_rate
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.patch_size = patch_size

        # Pose配置
        self.pose_x_vocab_size = pose_x_vocab_size
        self.pose_y_vocab_size = pose_y_vocab_size
        self.pose_z_vocab_size = pose_z_vocab_size
        self.yaw_vocab_size = yaw_vocab_size
        self.yaw_pose_emb_dim = yaw_pose_emb_dim

        # RoPE 配置
        self.use_rope = use_rope
        self.rope_theta = rope_theta


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * config.embed_dim, config.embed_dim, bias=False)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ==========================================
# 2. 轨迹时序编码器
# ==========================================
class TrajectoryTemporalEncoder(nn.Module):
    """
    负责将高频的轨迹特征 (T_raw) 投影并降采样到 (t_latent)。
    处理流程： Fourier -> MLP Projection -> Temporal Conv1d Downsample
    """
    def __init__(self, in_dim, out_dim, compression_rate=4):
        super().__init__()
        self.compression_rate = compression_rate

        self.projector = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=False),
            nn.LayerNorm(out_dim)
        )

        if compression_rate > 1:
            self.downsampler = nn.Conv1d(
                in_channels=out_dim,
                out_channels=out_dim,
                kernel_size=compression_rate,
                stride=compression_rate
            )
        else:
            self.downsampler = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B_Nc, T_raw, 4, In_Dim]
        Returns:
            x_out: [B_Nc, t_latent, 4, Out_Dim]
        """
        B_Nc, T_raw, Num_Comps, In_Dim = x.shape

        x = self.projector(x)
        x = rearrange(x, 'b t k c -> (b k) c t')
        x = self.downsampler(x)
        x = rearrange(x, '(b k) c t -> b t k c', k=Num_Comps)

        return x


# ==========================================
# 3. Attention 模块 (支持 RoPE)
# ==========================================
class GeneralSelfAttention(nn.Module):
    def __init__(self, config, is_causal=False, use_rope=False):
        super().__init__()
        assert config.embed_dim % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.embed_dim // config.n_head
        self.is_causal = is_causal
        self.use_rope = use_rope

        self.key = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.query = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.value = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

    def forward(self, x, freqs_cos=None, freqs_sin=None):
        """
        Args:
            x: [B, Seq, C]
            freqs_cos, freqs_sin: [Seq, head_dim] (optional, for RoPE)
        """
        B, Seq, C = x.size()

        # Project to Q, K, V: [B, Seq, n_head, head_dim]
        q = self.query(x).view(B, Seq, self.n_head, self.head_dim)
        k = self.key(x).view(B, Seq, self.n_head, self.head_dim)
        v = self.value(x).view(B, Seq, self.n_head, self.head_dim)

        # Transpose to [B, n_head, Seq, head_dim] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE if enabled
        if self.use_rope and freqs_cos is not None and freqs_sin is not None:
            q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        # Flash Attention
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=self.is_causal
        )

        y = y.transpose(1, 2).contiguous().view(B, Seq, C)
        y = self.resid_drop(self.proj(y))
        return y


# ==========================================
# 4. Transformer Blocks (支持 RoPE)
# ==========================================
class TimeBlock(nn.Module):
    """时间维度 Attention (使用 1D RoPE)"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = GeneralSelfAttention(config, is_causal=True, use_rope=config.use_rope)
        self.mlp = MLP(config)

    def forward(self, x, Nc, T, L, rope_1d=None):
        """
        Args:
            x: [(B*Nc), T, L, C]
            rope_1d: (freqs_cos, freqs_sin) 每个是 [T, head_dim]
        """
        residual = x
        x = self.ln1(x)
        # Rearrange: [(B*Nc), T, L, C] -> [(B*Nc*L), T, C]
        x = rearrange(x, 'b_nc t l c -> (b_nc l) t c')

        # Apply 1D RoPE for time dimension
        if rope_1d is not None:
            x = self.attn(x, freqs_cos=rope_1d[0], freqs_sin=rope_1d[1])
        else:
            x = self.attn(x)

        x = rearrange(x, '(b_nc l) t c -> b_nc t l c', l=L)
        x = residual + x

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        return x


class SpaceBlock(nn.Module):
    """空间维度 Attention (使用 2D RoPE)"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = GeneralSelfAttention(config, is_causal=False, use_rope=config.use_rope)
        self.mlp = MLP(config)

    def forward(self, x, Nc, T, L, rope_2d=None):
        """
        Args:
            x: [(B*Nc), T, L, C]
            rope_2d: (freqs_cos, freqs_sin) 每个是 [L, head_dim]
        """
        residual = x
        x = self.ln1(x)
        # Rearrange: [(B*Nc), T, L, C] -> [(B*Nc*T), L, C]
        x = rearrange(x, 'b_nc t l c -> (b_nc t) l c')

        # Apply 2D RoPE for space dimension
        if rope_2d is not None:
            x = self.attn(x, freqs_cos=rope_2d[0], freqs_sin=rope_2d[1])
        else:
            x = self.attn(x)

        x = rearrange(x, '(b_nc t) l c -> b_nc t l c', t=T)
        x = residual + x

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        return x


class CrossViewBlock(nn.Module):
    """跨视角 Attention (不使用 RoPE)"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = GeneralSelfAttention(config, is_causal=False, use_rope=False)
        self.mlp = MLP(config)

    def forward(self, x, Nc, T, L):
        residual = x
        x = self.ln1(x)
        # Rearrange: [(B*Nc), T, L, C] -> [(B*T*L), Nc, C]
        x = rearrange(x, '(b nc) t l c -> (b t l) nc c', nc=Nc)
        x = self.attn(x)
        x = rearrange(x, '(b t l) nc c -> (b nc) t l c', t=T, l=L)
        x = residual + x

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        return x


class AggregatorLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.time_block = TimeBlock(config)
        self.space_block = SpaceBlock(config)
        self.view_block = CrossViewBlock(config)

    def forward(self, x, Nc, T, L, rope_1d=None, rope_2d=None):
        x = self.time_block(x, Nc, T, L, rope_1d=rope_1d)
        x = self.space_block(x, Nc, T, L, rope_2d=rope_2d)
        x = self.view_block(x, Nc, T, L)
        return x


# ==========================================
# 5. 主模型: MVLatentTransformer
# ==========================================
class MVLatentTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 3D Patch Embedding
        self.patch_size = getattr(config, 'patch_size', (1, 2, 2))
        self.embedder_3d = PatchEmbed3D(
            patch_size=self.patch_size,
            in_chans=config.input_vae_dim,
            embed_dim=config.embed_dim,
            norm_layer=None,
            flatten=True,
        )

        # Trajectory Processing
        self.yaw_pose_emb_dim = config.yaw_pose_emb_dim
        self.yaw_vocab_num = config.yaw_vocab_size
        self.pose_x_vocab_num = config.pose_x_vocab_size
        self.pose_y_vocab_num = config.pose_y_vocab_size
        self.pose_z_vocab_num = config.pose_z_vocab_size

        self.traj_encoder = TrajectoryTemporalEncoder(
            in_dim=self.yaw_pose_emb_dim,
            out_dim=config.embed_dim,
            compression_rate=config.time_compression_rate
        )

        # Learnable Positional Embeddings
        t_patch = config.latent_size[0] // self.patch_size[0]
        h_patch = config.latent_size[1] // self.patch_size[1]
        w_patch = config.latent_size[2] // self.patch_size[2]

        self.pos_emb_time = nn.Parameter(torch.randn(1, 1, t_patch, 1, config.embed_dim) * 0.02)
        self.pos_emb_space = nn.Parameter(torch.randn(1, 1, 1, h_patch * w_patch, config.embed_dim) * 0.02)
        self.pos_emb_view = nn.Parameter(torch.randn(1, config.n_views, 1, 1, config.embed_dim) * 0.02)

        # RoPE Embeddings (预计算，注册为 buffer)
        if config.use_rope:
            head_dim = config.embed_dim // config.n_head

            # 1D RoPE for Time
            rope_1d_cos, rope_1d_sin = get_1d_rotary_pos_embed(head_dim, t_patch, config.rope_theta)
            self.register_buffer('rope_1d_cos', rope_1d_cos)
            self.register_buffer('rope_1d_sin', rope_1d_sin)

            # 2D RoPE for Space
            rope_2d_cos, rope_2d_sin = get_2d_rotary_pos_embed(head_dim, h_patch, w_patch, config.rope_theta)
            self.register_buffer('rope_2d_cos', rope_2d_cos)
            self.register_buffer('rope_2d_sin', rope_2d_sin)
        else:
            self.rope_1d_cos = None
            self.rope_1d_sin = None
            self.rope_2d_cos = None
            self.rope_2d_sin = None

        # Transformer Blocks
        self.layers = nn.ModuleList([
            AggregatorLayer(config) for _ in range(config.n_layer)
        ])

        self.final_norm = nn.LayerNorm(config.embed_dim)

    def get_raw_fourier_features(self, pose_indices, yaw_indices):
        """将索引转换为 Fourier Embeddings"""
        yaw_norm = yaw_indices / self.yaw_vocab_num
        pose_x_norm = pose_indices[..., 0:1] / self.pose_x_vocab_num
        pose_y_norm = pose_indices[..., 1:2] / self.pose_y_vocab_num
        pose_z_norm = pose_indices[..., 2:3] / self.pose_z_vocab_num

        coords = torch.cat([yaw_norm, pose_x_norm, pose_y_norm, pose_z_norm], dim=-1)
        fourier_emb = get_fourier_embeds_from_coordinates(self.yaw_pose_emb_dim, coords)
        return fourier_emb

    def get_pose_tokens(self, pose_indices, yaw_indices):
        """处理轨迹: 高频索引 -> Fourier -> 降采样"""
        B_Nc, T_raw, _ = pose_indices.shape

        raw_fourier = self.get_raw_fourier_features(pose_indices, yaw_indices)
        traj_tokens = self.traj_encoder(raw_fourier)
        traj_tokens = rearrange(traj_tokens, 'b t k c -> (b t) k c')

        return traj_tokens

    def forward(self, x):
        """
        Args:
            x: (B*Nc, C, t, h, w) - Raw VAE Output
        Returns:
            x_out: [(B*Nc*t_patch), L, C]
        """
        Nc = self.config.n_views

        # 3D Patch Embedding
        x = self.embedder_3d(x)  # [(B*Nc), t_patch, h_patch*w_patch, C]
        B_Nc, T, L, C = x.shape

        # Add Learnable Positional Embeddings
        x = rearrange(x, '(b nc) t l c -> b nc t l c', nc=Nc)
        x = x + self.pos_emb_time + self.pos_emb_space + self.pos_emb_view
        x = rearrange(x, 'b nc t l c -> (b nc) t l c')

        # Prepare RoPE
        rope_1d = (self.rope_1d_cos, self.rope_1d_sin) if self.config.use_rope else None
        rope_2d = (self.rope_2d_cos, self.rope_2d_sin) if self.config.use_rope else None

        # Pass through Transformer blocks
        for layer in self.layers:
            x = layer(x, Nc, T, L, rope_1d=rope_1d, rope_2d=rope_2d)

        x = self.final_norm(x)
        x_out = rearrange(x, 'b_nc t l c -> (b_nc t) l c')

        return x_out


# ==========================================
# 6. 测试验证
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("Testing MVLatentTransformer with 1D/2D RoPE")
    print("="*60)

    # 配置
    B_real = 2
    Nc = 6
    C_vae = 32

    t_latent, h, w = 4, 32, 32
    time_compression = 4
    T_raw = t_latent * time_compression

    embed_dim = 1024

    config = MVConfig(
        input_vae_dim=C_vae,
        embed_dim=embed_dim,
        n_views=Nc,
        n_layer=2,
        latent_size=(t_latent, h, w),
        time_compression_rate=time_compression,
        use_rope=True,
    )

    model = MVLatentTransformer(config).cuda()

    print(f"\n>>> 配置:")
    print(f"    VAE Latent: T={t_latent}, H={h}, W={w}")
    print(f"    Patch Size: {config.patch_size}")
    print(f"    After Patch: T={t_latent//config.patch_size[0]}, H={h//config.patch_size[1]}, W={w//config.patch_size[2]}")
    print(f"    Use RoPE: {config.use_rope}")
    print(f"    Rope 1D shape: {model.rope_1d_cos.shape if model.rope_1d_cos is not None else None}")
    print(f"    Rope 2D shape: {model.rope_2d_cos.shape if model.rope_2d_cos is not None else None}")

    # Test 1: Pose Tokens
    print("\n" + "="*60)
    print("Test 1: Trajectory Encoding")
    print("="*60)

    total_batch = B_real * Nc
    pose_indices = torch.randint(0, 512, (total_batch, T_raw, 3)).float().cuda()
    yaw_indices = torch.randint(0, 512, (total_batch, T_raw, 1)).float().cuda()

    pose_tokens = model.get_pose_tokens(pose_indices, yaw_indices)

    expected_pose_shape = (total_batch * t_latent, 4, embed_dim)
    print(f"Input:  Pose indices {pose_indices.shape}, Yaw indices {yaw_indices.shape}")
    print(f"Output: Pose tokens {pose_tokens.shape}")
    print(f"Expected: {expected_pose_shape}")
    assert pose_tokens.shape == expected_pose_shape
    print("✅ Trajectory Encoding Passed")

    # Test 2: Feature Aggregation with RoPE
    print("\n" + "="*60)
    print("Test 2: Feature Aggregation with RoPE")
    print("="*60)

    vae_output = torch.randn(total_batch, C_vae, t_latent, h, w).cuda()
    feature_output = model(vae_output)

    expect_t_latent = t_latent // config.patch_size[0]
    expected_hw = (h // config.patch_size[1]) * (w // config.patch_size[2])
    expected_feat_shape = (total_batch * expect_t_latent, expected_hw, embed_dim)

    print(f"Input:  VAE output {vae_output.shape}")
    print(f"Output: Features {feature_output.shape}")
    print(f"Expected: {expected_feat_shape}")
    assert feature_output.shape == expected_feat_shape
    print("✅ Feature Aggregation Passed")

    # Test 3: Integration
    print("\n" + "="*60)
    print("Test 3: Final Integration")
    print("="*60)

    final_input = torch.cat([pose_tokens, feature_output], dim=1)
    expected_final_shape = (total_batch * expect_t_latent, 4 + expected_hw, embed_dim)

    print(f"Pose tokens: {pose_tokens.shape}")
    print(f"Features:    {feature_output.shape}")
    print(f"Concat:      {final_input.shape}")
    print(f"Expected:    {expected_final_shape}")
    assert final_input.shape == expected_final_shape
    print("✅ Integration Passed")

    print("\n" + "="*60)
    print("🎉 All Tests Passed!")
    print("="*60)
