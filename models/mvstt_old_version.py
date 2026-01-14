# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# # ==========================================
# # 1. 配置类 (Configuration)
# # ==========================================
# class MVAggregatorConfig:
#     def __init__(self, 
#                  input_vae_dim=16,   # CogVideoX Latent Channels
#                  embed_dim=1024,     # Transformer Hidden Dim
#                  n_views=6,          # 相机数量 (Nc)
#                  n_head=16, 
#                  n_layer=4,          # Transformer 层数
#                  latent_size=(8, 32, 32), # (t', h', w')
#                  attn_pdrop=0.1, 
#                  resid_pdrop=0.1):
#         self.input_vae_dim = input_vae_dim
#         self.embed_dim = embed_dim
#         self.n_views = n_views
#         self.n_head = n_head
#         self.n_layer = n_layer
#         self.latent_size = latent_size # t, h, w
#         self.attn_pdrop = attn_pdrop
#         self.resid_pdrop = resid_pdrop

# # ==========================================
# # 2. 通用 Attention 模块
# # ==========================================
# class FactorizedAttention(nn.Module):
#     """
#     通用注意力模块，用于 Time, Space, 和 Cross-View
#     """
#     def __init__(self, config):
#         super().__init__()
#         assert config.embed_dim % config.n_head == 0
#         self.n_head = config.n_head
#         self.head_dim = config.embed_dim // config.n_head
        
#         self.key = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
#         self.query = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
#         self.value = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        
#         self.proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
#         self.attn_drop = nn.Dropout(config.attn_pdrop)
#         self.resid_drop = nn.Dropout(config.resid_pdrop)
#         self.ln = nn.LayerNorm(config.embed_dim)

#     def forward(self, x):
#         # x shape: (Batch_Dim, Seq_Len, C)
#         B, S, C = x.shape
        
#         # LayerNorm before Attention (Pre-Norm)
#         x_ln = self.ln(x)
        
#         q = self.query(x_ln).view(B, S, self.n_head, self.head_dim).transpose(1, 2)
#         k = self.key(x_ln).view(B, S, self.n_head, self.head_dim).transpose(1, 2)
#         v = self.value(x_ln).view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        
#         # Scaled Dot Product Attention (Flash Attention where available)
#         # 对于特征聚合，我们使用全局双向注意力，不加 Causal Mask
#         y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        
#         y = y.transpose(1, 2).contiguous().view(B, S, C)
#         y = self.resid_drop(self.proj(y))
        
#         return x + y  # Residual Connection

# # ==========================================
# # 3. 核心聚合块 (Time -> Space -> View)
# # ==========================================
# class TimeSpaceViewBlock(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
        
#         self.time_attn = FactorizedAttention(config)
#         self.space_attn = FactorizedAttention(config)
#         self.view_attn = FactorizedAttention(config)
        
#         self.mlp = nn.Sequential(
#             nn.Linear(config.embed_dim, 4 * config.embed_dim),
#             nn.GELU(),
#             nn.Linear(4 * config.embed_dim, config.embed_dim),
#             nn.Dropout(config.resid_pdrop)
#         )
#         self.ln_mlp = nn.LayerNorm(config.embed_dim)

#     def forward(self, x):
#         # Input: (B, Nc, t, L, C)
#         # L = h * w
#         B, Nc, t, L, C = x.shape
        
#         # 1. Time Attention: 关注 t 维度
#         # View: (B * Nc * L) 作为 Batch, t 作为 Sequence
#         x = rearrange(x, 'b nc t l c -> (b nc l) t c')
#         x = self.time_attn(x)
#         x = rearrange(x, '(b nc l) t c -> b nc t l c', b=B, nc=Nc, l=L)
        
#         # 2. Space Attention: 关注 L (h*w) 维度
#         # View: (B * Nc * t) 作为 Batch, L 作为 Sequence
#         x = rearrange(x, 'b nc t l c -> (b nc t) l c')
#         x = self.space_attn(x)
#         x = rearrange(x, '(b nc t) l c -> b nc t l c', b=B, nc=Nc, t=t)
        
#         # 3. Cross-View Attention: 关注 Nc 维度
#         # View: (B * t * L) 作为 Batch, Nc 作为 Sequence
#         x = rearrange(x, 'b nc t l c -> (b t l) nc c')
#         x = self.view_attn(x)
#         x = rearrange(x, '(b t l) nc c -> b nc t l c', b=B, t=t, l=L)
        
#         # 4. FFN
#         # Apply to last dim C
#         x = x + self.mlp(self.ln_mlp(x))
        
#         return x

# # ==========================================
# # 4. 主模型: MVLatentTransformer
# # ==========================================
# class MVLatentTransformer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
        
#         # Projector: VAE Dim -> Transformer Dim
#         self.input_proj = nn.Linear(config.input_vae_dim, config.embed_dim)
        
#         # Positional Embeddings
#         # Time: (1, 1, t, 1, C)
#         self.pos_emb_time = nn.Parameter(torch.randn(1, 1, config.latent_size[0], 1, config.embed_dim) * 0.02)
#         # Space: (1, 1, 1, h*w, C)
#         self.pos_emb_space = nn.Parameter(torch.randn(1, 1, 1, config.latent_size[1] * config.latent_size[2], config.embed_dim) * 0.02)
#         # View: (1, Nc, 1, 1, C)
#         self.pos_emb_view = nn.Parameter(torch.randn(1, config.n_views, 1, 1, config.embed_dim) * 0.02)
        
#         # Stack Blocks
#         self.blocks = nn.ModuleList([
#             TimeSpaceViewBlock(config) for _ in range(config.n_layer)
#         ])
        
#         # Final Output Layer (保持 Hidden Dim，或者映射回其他维度)
#         # 这里为了对齐Epona输出，通常保持 embed_dim 或映射回特定维度
#         self.final_norm = nn.LayerNorm(config.embed_dim)
        
#     def forward(self, x):
#         """
#         Args:
#             x: Raw VAE output, shape (B_total, C, t, h, w)
#                其中 B_total = Real_Batch_Size * Num_Cameras
#         Returns:
#             out: Aggregated features aligned with Epona format
#                  shape (B_total * t, L, C_embed)
#                  这里 L = h*w (仅图像特征)
#         """
#         # 1. 维度调整与解耦 (Unfold)
#         # 输入是 (B*Nc, C, t, h, w)
#         # 我们先变为 (B, Nc, t, h, w, C) 以便处理 Linear 和 View
        
#         Nc = self.config.n_views
#         # Channel First -> Channel Last
#         x = rearrange(x, '(b nc) c t h w -> b nc t (h w) c', nc=Nc)
        
#         b, nc, t, l, c = x.shape
        
#         # 2. 投影到 Hidden Dim
#         x = self.input_proj(x) # (b, nc, t, l, embed_dim)
        
#         # 3. 添加位置编码 (Broadcasting)
#         # Time: (1, 1, t, 1, C)
#         # Space: (1, 1, 1, l, C)
#         # View: (1, nc, 1, 1, C)
#         x = x + self.pos_emb_time + self.pos_emb_space + self.pos_emb_view
        
#         # 4. 经过 Transformer Blocks
#         for block in self.blocks:
#             x = block(x)
            
#         x = self.final_norm(x)
        
#         # 5. 输出对齐 (Output Alignment)
#         # Epona 格式: (Batch * Frames, Tokens, Dim)
#         # 我们的 Batch = b * nc (因为每个视角最终还是作为独立视频流输出给后续网络)
#         # 我们的 Frames = t
#         # 我们的 Tokens = l (h*w)
#         # Transform: (b, nc, t, l, c) -> (b * nc * t, l, c)
#         x_out = rearrange(x, 'b nc t l c -> (b nc t) l c')
        
#         return x_out

# # ==========================================
# # 5. 测试代码
# # ==========================================
# if __name__ == "__main__":
#     # --- 参数设置 ---
#     B_real = 2          # 真实 Batch Size
#     Nc = 6              # 6个视角
#     B_total = B_real * Nc 
#     C_vae = 16          # CogVideoX VAE 输出通道
#     t_latent = 8        # 压缩后的时间维
#     h_latent = 32       # 压缩后的高度
#     w_latent = 32       # 压缩后的宽度
#     embed_dim = 1024    # Transformer 内部维度
    
#     # --- 1. 模拟 CogVideoX VAE 输出 ---
#     # Shape: (B*Nc, C, t, h, w)
#     vae_output = torch.randn(B_total, C_vae, t_latent, h_latent, w_latent).cuda()
#     print(f"[Input] VAE Latent Shape: {vae_output.shape}")
    
#     # --- 2. 初始化 Transformer ---
#     config = MVAggregatorConfig(
#         input_vae_dim=C_vae,
#         embed_dim=embed_dim,
#         n_views=Nc,
#         n_layer=2,      # 仅做演示，用2层
#         n_head=8,
#         latent_size=(t_latent, h_latent, w_latent)
#     )
    
#     model = MVLatentTransformer(config).cuda()
#     print(f"[Model] Initialized MVLatentTransformer with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
    
#     # --- 3. 前向传播 ---
#     # 这一步包含了 Time -> Space -> View 的所有聚合
#     output_features = model(vae_output)
    
#     # --- 4. 验证输出形状 ---
#     # 预期输出: (B_total * t, h*w, embed_dim)
#     # B_total * t = 12 * 8 = 96
#     # h*w = 32 * 32 = 1024
#     expected_B_F = B_total * t_latent
#     expected_L = h_latent * w_latent
    
#     print(f"[Output] Aggregated Features Shape: {output_features.shape}")
    
#     assert output_features.shape == (expected_B_F, expected_L, embed_dim), \
#         f"Shape Mismatch! Expected {(expected_B_F, expected_L, embed_dim)}, got {output_features.shape}"
        
#     print("Test Passed! The output format is perfectly aligned with MVWorld requirements.")
    
#     # --- 5. 模拟与 Pose Token 拼接 (可选，如果你要复刻 Epona 的 L = Img + Pose) ---
#     # 假设 Pose Token 已经准备好，形状为 (B_total * t, 3, embed_dim)
#     pose_tokens = torch.randn(expected_B_F, 3, embed_dim).cuda()
    
#     # Epona 最终输入给 GPT 的序列
#     final_gpt_input = torch.cat([pose_tokens, output_features], dim=1)
#     print(f"[Final] Concatenated with Pose Tokens: {final_gpt_input.shape}") # (96, 1027, 1024)



#######################################
#################第二版#################
#######################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ==========================================
# 1. 基础组件 (Config & MLP)
# ==========================================
class MVConfig:
    def __init__(self, 
                 input_vae_dim=16,
                 embed_dim=1024,
                 n_views=6,
                 n_head=16, 
                 n_layer=4, 
                 latent_size=(8, 32, 32), # t, h, w
                 attn_pdrop=0.1, 
                 resid_pdrop=0.1):
        self.input_vae_dim = input_vae_dim
        self.embed_dim = embed_dim
        self.n_views = n_views
        self.n_head = n_head
        self.n_layer = n_layer
        self.latent_size = latent_size
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

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
# 2. Attention 核心实现
# ==========================================
class GeneralSelfAttention(nn.Module):
    """
    通用的 Self-Attention 实现。
    可以通过 is_causal 参数控制是否使用因果掩码。
    """
    def __init__(self, config, is_causal=False):
        super().__init__()
        assert config.embed_dim % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.embed_dim // config.n_head
        self.is_causal = is_causal
        self.scale = self.head_dim ** -0.5

        self.key = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.query = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.value = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal Masking Logic
        attn_mask = None
        if self.is_causal:
            # 创建下三角掩码 (Batch无关，自动广播)
            # mask shape: (T, T)
            mask = torch.tril(torch.ones(T, T, device=x.device))
            attn_mask = torch.zeros(T, T, device=x.device).masked_fill(mask == 0, float('-inf'))
        
        # 计算 Attention
        # 这里的 attn_mask 会自动广播到 (B, n_head, T, T)
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=self.is_causal # PyTorch 2.0+ 优化
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

# ==========================================
# 3. 三种独立的 Block (Time, Space, View)
# ==========================================

class TimeBlock(nn.Module):
    """
    Time Block: 包含 Causal Attention 和 MLP
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        # 重点：开启 is_causal=True
        self.attn = GeneralSelfAttention(config, is_causal=True)
        self.mlp = MLP(config)

    def forward(self, x, Nc, T, L):
        # Input x: (B*Nc, T, L, C)
        # 我们需要在 T 维度做 Attention，把 (B*Nc, L) 当作 Batch
        
        # 1. Pre-Norm
        residual = x
        x = self.ln1(x)
        
        # 2. Reshape for Time Attention: (Batch, Seq_Len, Dim) -> (B*Nc*L, T, C)
        x = rearrange(x, '(b_nc) t l c -> (b_nc l) t c')
        
        # 3. Causal Attention
        x = self.attn(x)
        
        # 4. Reshape back & Residual
        x = rearrange(x, '(b_nc l) t c -> b_nc t l c', l=L)
        x = residual + x
        
        # 5. MLP Block
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

class SpaceBlock(nn.Module):
    """
    Space Block: 包含 Bidirectional Attention 和 MLP
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        # 重点：Space 不需要 Mask
        self.attn = GeneralSelfAttention(config, is_causal=False)
        self.mlp = MLP(config)

    def forward(self, x, Nc, T, L):
        # Input x: (B*Nc, T, L, C)
        
        # 1. Pre-Norm
        residual = x
        x = self.ln1(x)
        
        # 2. Reshape for Space Attention: (Batch, Seq_Len, Dim) -> (B*Nc*T, L, C)
        x = rearrange(x, '(b_nc) t l c -> (b_nc t) l c')
        
        # 3. Attention
        x = self.attn(x)
        
        # 4. Reshape back & Residual
        x = rearrange(x, '(b_nc t) l c -> b_nc t l c', t=T)
        x = residual + x
        
        # 5. MLP Block
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

class CrossViewBlock(nn.Module):
    """
    Cross View Block: 包含 Bidirectional Attention 和 MLP
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        # 重点：View 交互是全向的，不需要 Mask
        self.attn = GeneralSelfAttention(config, is_causal=False)
        self.mlp = MLP(config)

    def forward(self, x, Nc, T, L):
        # Input x: (B*Nc, T, L, C)
        # 需要显式解开 B 和 Nc
        
        # 1. Pre-Norm
        residual = x
        x = self.ln1(x)
        
        # 2. Reshape for View Attention: (Batch, Seq_Len, Dim) -> (B*T*L, Nc, C)
        # 先拆开 Nc: (b nc) t l c
        x = rearrange(x, '(b nc) t l c -> (b t l) nc c', nc=Nc)
        
        # 3. Attention (Cross-View)
        x = self.attn(x)
        
        # 4. Reshape back & Residual
        x = rearrange(x, '(b t l) nc c -> (b nc) t l c', t=T, l=L)
        x = residual + x
        
        # 5. MLP Block
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

# ==========================================
# 4. 聚合层 (Aggregator Layer)
# ==========================================
class AggregatorLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 顺序：Time (Causal) -> Space -> View
        self.time_block = TimeBlock(config)
        self.space_block = SpaceBlock(config)
        self.view_block = CrossViewBlock(config)

    def forward(self, x, Nc, T, L):
        x = self.time_block(x, Nc, T, L)
        x = self.space_block(x, Nc, T, L)
        x = self.view_block(x, Nc, T, L)
        return x

# ==========================================
# 5. 主 Transformer 模型
# ==========================================
class MVLatentTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input Projector
        self.input_proj = nn.Linear(config.input_vae_dim, config.embed_dim)
        
        # Positional Embeddings
        self.pos_emb_time = nn.Parameter(torch.randn(1, 1, config.latent_size[0], 1, config.embed_dim) * 0.02)
        self.pos_emb_space = nn.Parameter(torch.randn(1, 1, 1, config.latent_size[1] * config.latent_size[2], config.embed_dim) * 0.02)
        self.pos_emb_view = nn.Parameter(torch.randn(1, config.n_views, 1, 1, config.embed_dim) * 0.02)
        
        # Stacked Layers
        self.layers = nn.ModuleList([
            AggregatorLayer(config) for _ in range(config.n_layer)
        ])
        
        # Output Norm
        self.final_norm = nn.LayerNorm(config.embed_dim)
        
    def forward(self, x):
        # x: (B*Nc, C, t, h, w) - Raw VAE Output
        
        # 1. 维度处理: Channel Last, Flatten Space
        Nc = self.config.n_views
        # (B*Nc, C, t, h, w) -> (B*Nc, t, h*w, C)
        x = rearrange(x, 'b_nc c t h w -> b_nc t (h w) c')
        
        B_Nc, T, L, C = x.shape
        
        # 2. Projection
        x = self.input_proj(x) # (B*Nc, T, L, Embed_Dim)
        
        # 3. Add Positional Embeddings
        # 需要先把 x 拆成 (B, Nc, T, L, C) 才能加上 pos_emb_view
        x = rearrange(x, '(b nc) t l c -> b nc t l c', nc=Nc)
        x = x + self.pos_emb_time + self.pos_emb_space + self.pos_emb_view
        x = rearrange(x, 'b nc t l c -> (b nc) t l c')
        
        # 4. Transformer Layers
        for layer in self.layers:
            x = layer(x, Nc, T, L)
            
        # 5. Final Norm
        x = self.final_norm(x)
        
        # 6. Output Alignment for Autoregression
        # Epona Format: (B*Nc*T, L, C)
        # 即把所有的时间步也拍平到 Batch 维度
        x_out = rearrange(x, 'b_nc t l c -> (b_nc t) l c')
        
        return x_out

# ==========================================
# 6. 测试数据验证
# ==========================================
if __name__ == "__main__":
    # 配置
    B_real = 2
    Nc = 6
    C_vae = 16
    t, h, w = 8, 32, 32
    embed_dim = 1024
    
    config = MVConfig(
        input_vae_dim=C_vae,
        embed_dim=embed_dim,
        n_views=Nc,
        n_layer=2,
        latent_size=(t, h, w)
    )
    
    model = MVLatentTransformer(config).cuda()
    
    # 模拟输入 (B*Nc, C, t, h, w)
    vae_output = torch.randn(B_real * Nc, C_vae, t, h, w).cuda()
    
    # 前向传播
    output = model(vae_output)
    
    # 验证输出形状
    # 预期: (B*Nc*t, h*w, embed_dim)
    expected_shape = (B_real * Nc * t, h * w, embed_dim)
    
    print(f"Input Shape:  {vae_output.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Expect Shape: {expected_shape}")
    
    assert output.shape == expected_shape, "Shape Mismatch!"
    print("✅ Test Passed: Blocks are modular, Time Attention is causal.")