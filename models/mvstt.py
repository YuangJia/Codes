import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# ==========================================
# 0. 辅助工具 (模拟 utils)
# ==========================================
def get_fourier_embeds_from_coordinates(embed_dim, coordinates, max_period: int = 100,):
    """
    Args:
        embed_dim: int
        coordinates: a tensor [B x N x C] representing the coordinates of N points in C dimensions
    Returns:
        [B x N x C x embed_dim] tensor of positional embeddings
    """
    half_embed_dim = embed_dim // 2
    B, N, C = coordinates.shape
    emb = max_period ** (torch.arange(half_embed_dim, dtype=torch.float32, device=coordinates.device) / half_embed_dim)
    emb = emb[None, None, None, :] * coordinates[:, :, :, None]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb

# ==========================================
# 1. 配置类
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
                 resid_pdrop=0.1,
                 # 词表配置
                 pose_x_vocab_size=512,
                 pose_y_vocab_size=512,
                 pose_z_vocab_size=512, # [新增 Z]
                 yaw_vocab_size=512,
                 yaw_pose_emb_dim=512):
        self.input_vae_dim = input_vae_dim
        self.embed_dim = embed_dim
        self.n_views = n_views
        self.n_head = n_head
        self.n_layer = n_layer
        self.latent_size = latent_size
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        
        # Pose配置
        self.pose_x_vocab_size = pose_x_vocab_size
        self.pose_y_vocab_size = pose_y_vocab_size
        self.pose_z_vocab_size = pose_z_vocab_size
        self.yaw_vocab_size = yaw_vocab_size
        self.yaw_pose_emb_dim = yaw_pose_emb_dim

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
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=self.is_causal
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
        
        # 1. Pre-Norm
        residual = x
        x = self.ln1(x)
        
        # 2. Reshape for View Attention: (Batch, Seq_Len, Dim) -> (B*T*L, Nc, C)
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
# 4. 聚合层
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
# 5. 主模型: MVLatentTransformer (带 Z 轴扩展)
# ==========================================
class MVLatentTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.C = config.embed_dim

        self.yaw_vocab_num = config.yaw_vocab_size
        self.pose_x_vocab_num = config.pose_x_vocab_size
        self.pose_y_vocab_num = config.pose_y_vocab_size
        self.pose_z_vocab_num = config.pose_z_vocab_size # 新增 Z
        self.yaw_pose_emb_dim = config.yaw_pose_emb_dim
        
        # -------------------------------------------
        # A. Feature Projector
        # -------------------------------------------
        self.input_proj = nn.Linear(config.input_vae_dim, config.embed_dim)
        
        # -------------------------------------------
        # B. Pose Projectors (集成 Z 轴)
        # -------------------------------------------
        self.yaw_pose_emb_dim = config.yaw_pose_emb_dim
        
        # X Projector
        self.pose_x_projector = nn.Sequential(
            nn.Linear(self.yaw_pose_emb_dim, self.yaw_pose_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.yaw_pose_emb_dim, self.C, bias=False),
            nn.LayerNorm(self.C)
        )
        # Y Projector
        self.pose_y_projector = nn.Sequential(
            nn.Linear(self.yaw_pose_emb_dim, self.yaw_pose_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.yaw_pose_emb_dim, self.C, bias=False),
            nn.LayerNorm(self.C)
        )
        # [新增] Z Projector
        self.pose_z_projector = nn.Sequential(
            nn.Linear(self.yaw_pose_emb_dim, self.yaw_pose_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.yaw_pose_emb_dim, self.C, bias=False),
            nn.LayerNorm(self.C)
        )
        # Yaw Projector
        self.yaw_projector = nn.Sequential(
            nn.Linear(self.yaw_pose_emb_dim, self.yaw_pose_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.yaw_pose_emb_dim, self.C, bias=False),
            nn.LayerNorm(self.C)
        )

        # -------------------------------------------
        # C. Positional Embeddings
        # -------------------------------------------
        self.pos_emb_time = nn.Parameter(torch.randn(1, 1, config.latent_size[0], 1, config.embed_dim) * 0.02)
        self.pos_emb_space = nn.Parameter(torch.randn(1, 1, 1, config.latent_size[1] * config.latent_size[2], config.embed_dim) * 0.02)
        self.pos_emb_view = nn.Parameter(torch.randn(1, config.n_views, 1, 1, config.embed_dim) * 0.02)
        
        # -------------------------------------------
        # D. Transformer Blocks
        # -------------------------------------------
        self.layers = nn.ModuleList([
            AggregatorLayer(config) for _ in range(config.n_layer)
        ])
        
        self.final_norm = nn.LayerNorm(config.embed_dim)
        
    def get_yaw_pose_emb(self, pose_indices, yaw_indices):
        # 情况 1: 没有 Pose，只有 Yaw
        if pose_indices == None:
            yaw_indices_normalize = yaw_indices / self.yaw_vocab_num
            yaw_emb = get_fourier_embeds_from_coordinates( 
                self.yaw_pose_emb_dim,
                yaw_indices_normalize)
            # 返回 4 个值，缺少的补 None
            return yaw_emb, None, None, None 

        # 情况 2: 只有 X (通常不太可能，但为了代码健壮性保留)
        elif pose_indices is not None and (pose_indices.shape[-1]==1):
            yaw_indices_normalize = yaw_indices / self.yaw_vocab_num
            pose_x_indices_normalize = pose_indices[:, :, 0:1] / self.pose_x_vocab_num
            
            yaw_pose_emb = get_fourier_embeds_from_coordinates(
                self.yaw_pose_emb_dim,
                torch.cat([yaw_indices_normalize, pose_x_indices_normalize], dim=-1), )
            
            yaw_emb, pose_x_emb = torch.split(yaw_pose_emb, dim=2, split_size_or_sections=1)
            # 返回 4 个值
            return yaw_emb, pose_x_emb, None, None

        # 情况 3: 完整的 Pose (X, Y, Z)
        else:
            # 1. 归一化 Yaw
            yaw_indices_normalize = yaw_indices / self.yaw_vocab_num
            
            # 2. 归一化 Pose X, Y, Z
            # 假设 pose_indices 的最后一维顺序是 [x, y, z]
            pose_x_indices_normalize = pose_indices[:, :, 0:1] / self.pose_x_vocab_num
            pose_y_indices_normalize = pose_indices[:, :, 1:2] / self.pose_y_vocab_num
            pose_z_indices_normalize = pose_indices[:, :, 2:3] / self.pose_z_vocab_num # [新增]
            
            # 3. 拼接并编码
            # concat 顺序: [yaw, x, y, z]
            coords = torch.cat([
                yaw_indices_normalize, 
                pose_x_indices_normalize, 
                pose_y_indices_normalize, 
                pose_z_indices_normalize # [新增]
            ], dim=-1)
            
            yaw_pose_emb = get_fourier_embeds_from_coordinates(
                self.yaw_pose_emb_dim,
                coords
            )
            
            # 4. 切分
            # 这里的 dim=2 对应的是 feature 维度 (大小为4)。
            # split 之后会得到 4 个形状为 [B, T, 1, Embed_Dim] 的张量
            yaw_emb, pose_x_emb, pose_y_emb, pose_z_emb = torch.split(yaw_pose_emb, dim=2, split_size_or_sections=1)
            
            return yaw_emb, pose_x_emb, pose_y_emb, pose_z_emb

    def forward(self, x):
        """
        主 Feature Aggregation 前向传播
        x: (B*Nc, C, t, h, w) - Raw VAE Output
        """
        # 1. 维度处理: Channel Last, Flatten Space
        Nc = self.config.n_views
        # (B*Nc, C, t, h, w) -> (B*Nc, t, h*w, C)
        x = rearrange(x, 'b_nc c t h w -> b_nc t (h w) c')
        
        B_Nc, T, L, C = x.shape
        
        # 2. Projection (Visual Features)
        x = self.input_proj(x) # (B*Nc, T, L, Embed_Dim)
        
        # 3. Add Positional Embeddings
        x = rearrange(x, '(b nc) t l c -> b nc t l c', nc=Nc)
        x = x + self.pos_emb_time + self.pos_emb_space + self.pos_emb_view
        x = rearrange(x, 'b nc t l c -> (b nc) t l c')
        
        # 4. Transformer Layers (Feature Aggregation)
        for layer in self.layers:
            x = layer(x, Nc, T, L)
            
        # 5. Final Norm
        x = self.final_norm(x)
        
        # 6. Output Alignment
        # Epona Format: (B*Nc*T, L, C)
        x_out = rearrange(x, 'b_nc t l c -> (b_nc t) l c')
        
        return x_out

    def get_pose_tokens(self, pose_indices, yaw_indices):
        """
        辅助函数：获取 Pose Tokens (包含 Z)，用于和 Transformer 输出拼接
        pose_indices: (B*Nc*T, 3) 已经展平的 indices
        """
        # 增加时间维以便复用 get_yaw_pose_emb
        if pose_indices.dim() == 2:
            pose_indices = pose_indices.unsqueeze(1)
            yaw_indices = yaw_indices.unsqueeze(1)
            
        yaw_emb, x_emb, y_emb, z_emb = self.get_yaw_pose_emb(pose_indices, yaw_indices)
        
        # Project
        t_yaw = self.yaw_projector(yaw_emb)
        t_x = self.pose_x_projector(x_emb)
        t_y = self.pose_y_projector(y_emb)
        t_z = self.pose_z_projector(z_emb) # [新增] Z Projector 调用
        
        # Concat: (B, 1, 4*C) -> reshape -> (B, 4, C)
        # 顺序: Yaw, X, Y, Z
        pose_tokens = torch.cat([t_yaw, t_x, t_y, t_z], dim=2)
        
        if pose_tokens.shape[1] == 1:
             pose_tokens = pose_tokens.squeeze(1) # (B, 4, C) if input was flattened
             # 但我们需要 (B, 4, C)，这里 4 是序列长度
             pose_tokens = pose_tokens.view(pose_tokens.shape[0], 4, -1)
             
        return pose_tokens

# ==========================================
# 6. 测试验证
# ==========================================
if __name__ == "__main__":
    # 配置
    B_real = 2
    Nc = 6
    C_vae = 16
    t, h, w = 4, 32, 32
    embed_dim = 1024
    
    config = MVConfig(
        input_vae_dim=C_vae,
        embed_dim=embed_dim,
        n_views=Nc,
        n_layer=1,
        latent_size=(t, h, w)
    )
    
    model = MVLatentTransformer(config).cuda()
    
    print(">>> 1. 测试 Feature Aggregation (Transformer部分)")
    # 模拟输入 (B*Nc, C, t, h, w)
    vae_output = torch.randn(B_real * Nc, C_vae, t, h, w).cuda()
    
    # 前向传播
    feature_output = model(vae_output)
    
    # 验证输出形状: (B*Nc*t, h*w, embed_dim)
    expected_feat_shape = (B_real * Nc * t, h * w, embed_dim)
    print(f"Feature Output: {feature_output.shape}")
    assert feature_output.shape == expected_feat_shape
    print("✅ Feature Aggregation Passed.")
    
    print("\n>>> 2. 测试 Pose Projection (包含 Z 轴)")
    # 模拟 Pose Indices: (Total_Frames, 3) -> x, y, z
    total_frames = B_real * Nc * t
    pose_indices = torch.randint(0, 512, (total_frames, 3)).float().cuda()
    yaw_indices = torch.randint(0, 512, (total_frames, 1)).float().cuda()
    
    pose_tokens = model.get_pose_tokens(pose_indices, yaw_indices)
    
    # 预期 Pose Token 形状: (Total_Frames, 4, embed_dim) -> 4 tokens (Yaw, X, Y, Z)
    print(f"Pose Tokens: {pose_tokens.shape}")
    assert pose_tokens.shape == (total_frames, 4, embed_dim)
    print("✅ Pose Projection (with Z) Passed.")
    
    print("\n>>> 3. 最终拼接 (模拟 Epona 输入)")
    # Epona 风格: concat([pose, features], dim=1)
    # L_final = 4 (Pose) + 1024 (Img) = 1028
    final_input = torch.cat([pose_tokens, feature_output], dim=1)
    print(f"Final Next DiT Input: {final_input.shape}")
    assert final_input.shape == (total_frames, 4 + h*w, embed_dim)
    print("✅ Integration Complete.")