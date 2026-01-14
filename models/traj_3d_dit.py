import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from einops import rearrange

from models.modules.dit_modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlockWithText,
    timestep_embedding,
)

# -----------------------------------------------------------------------------
# 2. 参数配置类
# -----------------------------------------------------------------------------
@dataclass
class TrajParams:
    in_channels: int
    out_channels: int
    context_in_dim: int     # Visual Latent Dim
    text_in_dim: int        # Text Encoder Dim (e.g. 768 for CLIP)
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool

# -----------------------------------------------------------------------------
# 3. 主模型: Tri-Modal TrajDiT
# -----------------------------------------------------------------------------
class TriModalTrajDiT(nn.Module):
    def __init__(self, params: TrajParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        # --- Embedders ---
        # 1. Trajectory
        self.traj_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        
        # 2. Time
        self.time_in = nn.Sequential(
            nn.Linear(256, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # 3. Visual Context
        self.cond_in = nn.Linear(params.context_in_dim, self.hidden_size)
        
        # 4. Text (NEW)
        # Project CLIP/T5 dimension to hidden_size
        self.text_in = nn.Linear(params.text_in_dim, self.hidden_size)
        
        # 5. PE
        pe_dim = params.hidden_size // params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim) # Assuming you have this from Epona code

        # --- Blocks ---
        # Phase 1: Dual Stream (Traj <-> Visual)
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=params.mlp_ratio,
                qkv_bias=params.qkv_bias,
            )
            for _ in range(params.depth)
        ])

        # Phase 2: Single Stream (Traj+Visual <-> Text)
        self.single_blocks = nn.ModuleList([
            SingleStreamBlockWithText(
                self.hidden_size, 
                self.num_heads, 
                mlp_ratio=params.mlp_ratio
            )
            for _ in range(params.depth_single_blocks)
        ])

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        traj: Tensor,          # (B, L_traj, 3)
        traj_ids: Tensor,      # PE IDs
        cond: Tensor,          # (B, L_vis, C_vis) - Visual
        cond_ids: Tensor,      # PE IDs
        text_tokens: Tensor,   # (B, L_txt, C_txt) - Text Sequence
        text_pooled: Tensor,   # (B, C_txt) - Global Text Vector
        timesteps: Tensor,     # (B,)
        guidance: Tensor | None = None,
        patch_size: list = [1, 2, 2],
    ) -> Tensor:
        
        # --- 1. Embedding ---
        traj = self.traj_in(traj)
        cond = self.cond_in(cond)
        
        # Text Embedding
        text_tokens = self.text_in(text_tokens) # (B, L_txt, H)
        text_pooled = self.text_in(text_pooled) # (B, H)

        # Time & Global Vector
        t_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(t_emb)
        
        # Fuse Global Text into Vector (AdaLN Control)
        vec = vec + text_pooled

        # PE Calculation (Simplified for demo)
        ids = torch.cat((cond_ids, traj_ids), dim=1)
        pe = self.pe_embedder(ids)

        # --- 2. Phase 1: Dual Stream (Physical Alignment) ---
        for block in self.double_blocks:
            # Traj interacts with Visual Context
            traj, cond = block(traj, cond=cond, vec=vec, pe=pe)

        # --- 3. Phase 2: Single Stream (Text Injection) ---
        # Concatenate Trajectory and Visual Context
        # This forms the "Physical World" sequence
        mixed_seq = torch.cat((cond, traj), dim=1)
        
        for block in self.single_blocks:
            # Cross-Attend to Text Tokens
            mixed_seq = block(
                x=mixed_seq, 
                vec=vec, 
                pe=pe, 
                text_tokens=text_tokens
            )
            
        # --- 4. Output ---
        # Extract Trajectory part (it was at the end)
        traj_out = mixed_seq[:, cond.shape[1]:, ...]
        
        traj_out = self.final_layer(traj_out, vec)
        return traj_out