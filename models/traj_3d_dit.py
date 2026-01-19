# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# from dataclasses import dataclass
# from einops import rearrange

# from models.modules.dit_modules.layers import (
#     DoubleStreamBlock,
#     EmbedND,
#     LastLayer,
#     MLPEmbedder,
#     SingleStreamBlockWithText,
#     timestep_embedding,
# )

# # -----------------------------------------------------------------------------
# # 2. 参数配置类
# # -----------------------------------------------------------------------------
# @dataclass
# class TrajParams:
#     in_channels: int
#     out_channels: int
#     context_in_dim: int     # Visual Latent Dim
#     text_in_dim: int        # Text Encoder Dim (e.g. 768 for CLIP)
#     hidden_size: int
#     mlp_ratio: float
#     num_heads: int
#     depth: int
#     depth_single_blocks: int
#     axes_dim: list[int]
#     theta: int
#     qkv_bias: bool
#     guidance_embed: bool

# # -----------------------------------------------------------------------------
# # 3. 主模型: Tri-Modal TrajDiT
# # -----------------------------------------------------------------------------
# class TriModalTrajDiT(nn.Module):
#     def __init__(self, params: TrajParams):
#         super().__init__()

#         self.params = params
#         self.in_channels = params.in_channels
#         self.out_channels = params.out_channels
#         self.hidden_size = params.hidden_size
#         self.num_heads = params.num_heads

#         # --- Embedders ---
#         # 1. Trajectory
#         self.traj_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        
#         # 2. Time
#         self.time_in = nn.Sequential(
#             nn.Linear(256, self.hidden_size),
#             nn.SiLU(),
#             nn.Linear(self.hidden_size, self.hidden_size)
#         )
        
#         # 3. Visual Context
#         self.cond_in = nn.Linear(params.context_in_dim, self.hidden_size)
        
#         # 4. Text (NEW)
#         # Project CLIP/T5 dimension to hidden_size
#         self.text_in = nn.Linear(params.text_in_dim, self.hidden_size)
        
#         # 5. PE
#         pe_dim = params.hidden_size // params.num_heads
#         self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim) # Assuming you have this from Epona code

#         # --- Blocks ---
#         # Phase 1: Dual Stream (Traj <-> Visual)
#         self.double_blocks = nn.ModuleList([
#             DoubleStreamBlock(
#                 self.hidden_size,
#                 self.num_heads,
#                 mlp_ratio=params.mlp_ratio,
#                 qkv_bias=params.qkv_bias,
#             )
#             for _ in range(params.depth)
#         ])

#         # Phase 2: Single Stream (Traj+Visual <-> Text)
#         self.single_blocks = nn.ModuleList([
#             SingleStreamBlockWithText(
#                 self.hidden_size, 
#                 self.num_heads, 
#                 mlp_ratio=params.mlp_ratio
#             )
#             for _ in range(params.depth_single_blocks)
#         ])

#         self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

#     def forward(
#         self,
#         traj: Tensor,          # (B, L_traj, 3)
#         traj_ids: Tensor,      # PE IDs
#         cond: Tensor,          # (B, L_vis, C_vis) - Visual
#         cond_ids: Tensor,      # PE IDs
#         text_tokens: Tensor,   # (B, L_txt, C_txt) - Text Sequence
#         text_pooled: Tensor,   # (B, C_txt) - Global Text Vector
#         timesteps: Tensor,     # (B,)
#         guidance: Tensor | None = None,
#         patch_size: list = [1, 2, 2],
#     ) -> Tensor:
        
#         # --- 1. Embedding ---
#         traj = self.traj_in(traj)
#         cond = self.cond_in(cond)
        
#         # Text Embedding
#         text_tokens = self.text_in(text_tokens) # (B, L_txt, H)
#         text_pooled = self.text_in(text_pooled) # (B, H)

#         # Time & Global Vector
#         t_emb = timestep_embedding(timesteps, 256)
#         vec = self.time_in(t_emb)
        
#         # Fuse Global Text into Vector (AdaLN Control)
#         vec = vec + text_pooled

#         # PE Calculation (Simplified for demo)
#         ids = torch.cat((cond_ids, traj_ids), dim=1)
#         pe = self.pe_embedder(ids)

#         # --- 2. Phase 1: Dual Stream (Physical Alignment) ---
#         for block in self.double_blocks:
#             # Traj interacts with Visual Context
#             traj, cond = block(traj, cond=cond, vec=vec, pe=pe)

#         # --- 3. Phase 2: Single Stream (Text Injection) ---
#         # Concatenate Trajectory and Visual Context
#         # This forms the "Physical World" sequence
#         mixed_seq = torch.cat((cond, traj), dim=1)
        
#         for block in self.single_blocks:
#             # Cross-Attend to Text Tokens
#             mixed_seq = block(
#                 x=mixed_seq, 
#                 vec=vec, 
#                 pe=pe, 
#                 text_tokens=text_tokens
#             )
            
#         # --- 4. Output ---
#         # Extract Trajectory part (it was at the end)
#         traj_out = mixed_seq[:, cond.shape[1]:, ...]
        
#         traj_out = self.final_layer(traj_out, vec)
#         return traj_out

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from einops import rearrange

# 复用 Epona 提供的 DiT 基础模块
from models.modules.dit_modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)

def mean_flat(tensor):
    """计算 MSE Loss 的平均值"""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

@dataclass
class Traj3DParams:
    """Physical DiT 的配置参数"""
    in_channels: int        # 这里将是: raw_dims * compression_rate
    out_channels: int       # 同上
    context_in_dim: int     # MVSTT 的 hidden_size (e.g., 1024)
    hidden_size: int        # DiT 内部 hidden_size (e.g., 1024 or 768)
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    # 新增物理参数
    raw_traj_dim: int = 3   # x, y, z
    compression_rate: int = 4 # 4帧合1

class Traj3DDiT(nn.Module):
    """
    Physical Trajectory Diffusion Transformer
    用于基于 MVSTT 的时空特征预测 3D 物理轨迹 (x, y, z)。
    """
    def __init__(self, params: Traj3DParams):
        super().__init__()
        self.params = params
        
        # 1. 维度计算逻辑 (关键修改点)
        # 输入通道 = 原始维度(3) * 时间压缩率(4) = 12
        # 这意味着 DiT 的每一个 Token 负责预测 4 帧的 XYZ 数据
        self.raw_dim = params.raw_traj_dim
        self.compression_rate = params.compression_rate
        expected_channels = self.raw_dim * self.compression_rate
        
        if params.in_channels != expected_channels:
            print(f"[Warning] Params in_channels ({params.in_channels}) != calculated ({expected_channels}). Using params value.")
        
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        # 2. 基础检查
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"Hidden size {self.hidden_size} must be divisible by num_heads {self.num_heads}")
        
        pe_dim = self.hidden_size // self.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")

        # 3. Embedding Layers
        # 位置编码 (用于区分 Traj Token 和 Cond Token)
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        
        # 轨迹输入映射: [N, 1, 12] -> [N, 1, 1024]
        self.traj_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        
        # 时间步编码
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        
        # Guidance (CFG)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        
        # 条件映射: MVSTT Output [N, L_img, 1024] -> [N, L_img, 1024]
        self.cond_in = nn.Linear(params.context_in_dim, self.hidden_size)

        # 4. Transformer Blocks (Flux Architecture)
        # Double Stream: Traj 和 Cond 独立交互
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=params.mlp_ratio,
                qkv_bias=params.qkv_bias,
            )
            for _ in range(params.depth)
        ])

        # Single Stream: Traj 和 Cond 混合交互
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
            for _ in range(params.depth_single_blocks)
        ])

        # 5. Output Layer
        # 输出维度回归到 12 (即预测 4 帧的 xyz)
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        traj: Tensor,       # [N, L_traj, C_in] (Noisy Trajectory)
        traj_ids: Tensor,   # [N, L_traj, 3] (Positional IDs)
        cond: Tensor,       # [N, L_img, C_ctx] (MVSTT Features)
        cond_ids: Tensor,   # [N, L_img, 3]
        timesteps: Tensor,  # [N] or [N, 1]
        guidance: Tensor | None = None,
    ) -> Tensor:
        """
        N = B * Nc * T_latent
        L_traj = 1 (通常每各 Latent Step 对应 1 个 Traj Token)
        """
        if traj.ndim != 3 or cond.ndim != 3:
            raise ValueError(f"Input traj ({traj.shape}) and cond ({cond.shape}) tensors must have 3 dimensions.")

        # 1. Embedding
        traj = self.traj_in(traj)   # [N, L_traj, D]
        cond = self.cond_in(cond)   # [N, L_img, D]
        
        # 2. Time & Guidance
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))

        # 3. Positional Embedding
        # Concat IDs along sequence dimension
        ids = torch.cat((cond_ids, traj_ids), dim=1) 
        pe = self.pe_embedder(ids)

        # 4. Double Blocks
        for block in self.double_blocks:
            traj, cond = block(traj, cond=cond, vec=vec, pe=pe)

        # 5. Single Blocks (Merge Streams)
        traj = torch.cat((cond, traj), 1)
        for block in self.single_blocks:
            traj = block(traj, vec=vec, pe=pe)
        
        # 6. Un-merge (Slice out trajectory part)
        traj = traj[:, cond.shape[1] :, ...]

        # 7. Final Projection
        traj = self.final_layer(traj, vec)  # [N, L_traj, C_out]
        return traj

    def training_losses(self, 
                        traj_gt_raw: Tensor,    # [B*Nc*T_raw, 1, 3] 原始高频轨迹 (XYZ)
                        traj_ids: Tensor,
                        cond: Tensor,           # [N, L_img, C] MVSTT Features
                        cond_ids: Tensor,
                        t: Tensor,              # Noise Level
                        guidance: Tensor | None = None,
                        noise: Tensor | None = None,
                    ) -> dict:
        """
        训练接口：负责数据折叠、加噪、计算 Loss
        """
        # 1. 数据预处理：时序折叠 (Temporal Stacking)
        # Input: [B*Nc, T_raw, 3] -> [B*Nc, T_latent, k, 3] -> [B*Nc*T_latent, 1, k*3]
        # 这里假设输入已经整理好 batch 为 N_raw，需要折叠成 N_latent
        
        # 假设 traj_gt_raw 是 [B_total, T_raw, 3]
        target_traj = self.stack_trajectory(traj_gt_raw) # -> [N, 1, 12]
        
        if noise is None:
            noise = torch.randn_like(target_traj)
        
        # 2. 加噪 (Diffusion Process)
        # t shape needs to broadcast: [N, 1, 1]
        if t.ndim == 1:
            t = t.view(-1, 1, 1)
            
        x_t = t * target_traj + (1. - t) * noise
        target_pred = target_traj - noise # Flow Matching Target (v-prediction)
        
        # 3. Forward Pass
        pred = self(
            traj=x_t, 
            traj_ids=traj_ids, 
            cond=cond, 
            cond_ids=cond_ids, 
            timesteps=t.reshape(-1), 
            guidance=guidance
        )
        
        # 4. Loss Calculation
        assert pred.shape == target_pred.shape, f"Pred {pred.shape} != Target {target_pred.shape}"
        
        # 预测的是 12 个单帧的数值 (k*3)
        mse = mean_flat((target_pred - pred) ** 2)
        
        terms = {
            "loss": mse.mean(),
            "mse": mse,
            "pred_raw": pred # 用于后续可视化 Debug
        }
        return terms

    def stack_trajectory(self, traj_raw: Tensor) -> Tensor:
        """
        将高频轨迹折叠为 DiT 输入格式
        Input:  [B_nc, T_raw, 3]
        Output: [B_nc * T_latent, 1, 12] (如果 k=4)
        """
        B_nc, T_raw, C = traj_raw.shape
        k = self.compression_rate
        T_latent = T_raw // k
        
        assert T_raw % k == 0, "Trajectory length must be divisible by compression rate"
        
        # Reshape: [B, t*k, c] -> [B, t, k, c] -> [B, t, 1, k*c]
        # 然后合并 Batch: [B*t, 1, k*c]
        traj_stacked = rearrange(traj_raw, 'b (t k) c -> (b t) 1 (k c)', k=k)
        return traj_stacked

    def unstack_trajectory(self, traj_stacked: Tensor, B_nc: int) -> Tensor:
        """
        将 DiT 输出还原为物理轨迹
        Input:  [B_nc * T_latent, 1, 12]
        Output: [B_nc, T_raw, 3]
        """
        k = self.compression_rate
        # [B*t, 1, k*c] -> [B, t, k*c] -> [B, t, k, c] -> [B, t*k, c]
        traj_raw = rearrange(traj_stacked, '(b t) 1 (k c) -> b (t k) c', b=B_nc, k=k)
        return traj_raw

    @torch.no_grad()
    def sample(self,
               traj_noise: Tensor, # [N, 1, 12]
               traj_ids: Tensor,
               cond: Tensor,
               cond_ids: Tensor,
               timesteps: list[float],
               B_nc: int,          # 用于最后还原形状
            ):
        """
        推理采样函数
        """
        traj = traj_noise
        
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((traj.shape[0],), t_curr, dtype=traj.dtype, device=traj.device)
            
            # Predict velocity/noise
            pred = self(
                traj=traj,
                traj_ids=traj_ids,
                cond=cond,
                cond_ids=cond_ids,
                timesteps=t_vec,
            )
            
            # Euler Step
            traj = traj + (t_prev - t_curr) * pred
            
        # 解开折叠，还原为 [B, T_raw, 3] 的 12个单帧序列
        traj_final = self.unstack_trajectory(traj, B_nc)
        return traj_final

# ==============================================================================
# Helper / Wrapper for usage (Optional)
# ==============================================================================
class PhysicalTrajWrapper(nn.Module):
    """
    一个简单的 Wrapper，用于封装 MVSTT 和 Traj3DDiT 的联合调用逻辑
    """
    def __init__(self, mvstt_model, traj_dit_model):
        super().__init__()
        self.mvstt = mvstt_model
        self.traj_dit = traj_dit_model
    
    def forward(self, 
                latents,        # [B*Nc, T, H, W] 
                traj_gt_raw,    # [B*Nc, T_raw, 3]
                traj_ids, 
                cond_ids, 
                t_noise):
        
        # 1. 获取 MVSTT 特征 (Condition)
        # Output: [B*Nc*T, L_img, 1024]
        stt_features = self.mvstt(latents) 
        
        # 2. 计算 Trajectory Loss
        # 内部会自动处理 Traj 的 reshape (stacking)
        loss_dict = self.traj_dit.training_losses(
            traj_gt_raw=traj_gt_raw,
            traj_ids=traj_ids,
            cond=stt_features,
            cond_ids=cond_ids,
            t=t_noise
        )
        
        return loss_dict