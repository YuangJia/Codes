# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# from dataclasses import dataclass
# from einops import rearrange
# import math

# # Import the DiT modules from your existing codebase
# from models.modules.dit_modules.layers import (
#     DoubleStreamBlock,
#     SingleStreamBlock,
#     timestep_embedding,
#     MLPEmbedder,
#     EmbedND,
#     LastLayer
# )


# def mean_flat(tensor):
#     """Average tensor over all dimensions except batch."""
#     return tensor.mean(dim=list(range(1, len(tensor.shape))))


# @dataclass
# class PhysicalDiTParams:
#     """Configuration for Physical DiT model."""
#     # Trajectory input/output configuration
#     traj_in_channels: int = 3  # xyz coordinates
#     traj_out_channels: int = 3  # predict xyz
    
#     # Model architecture
#     hidden_size: int = 1024
#     mlp_ratio: float = 4.0
#     num_heads: int = 16
#     depth: int = 8  # Number of DoubleStream blocks
#     depth_single_blocks: int = 16  # Number of SingleStream blocks
    
#     # Trajectory temporal downsampling
#     traj_temporal_compression: int = 4  # Match 3D VAE compression
#     T_origin: int = 16  # Original temporal length before compression
#     num_output_frames: int = 12  # Predict 12 single frames
    
#     # RoPE configuration
#     axes_dim: list[int] = None  # Will be set based on hidden_size
#     theta: int = 10000
#     qkv_bias: bool = True
    
#     # Guidance
#     guidance_embed: bool = True
    
#     def __post_init__(self):
#         if self.axes_dim is None:
#             # pe_dim must equal hidden_size // num_heads
#             pe_dim = self.hidden_size // self.num_heads
#             # For single temporal axis
#             self.axes_dim = [pe_dim]
        
#         # Validate temporal compression
#         if self.T_origin % self.traj_temporal_compression != 0:
#             raise ValueError(
#                 f"T_origin ({self.T_origin}) must be divisible by "
#                 f"traj_temporal_compression ({self.traj_temporal_compression})"
#             )


# class TrajectoryTemporalDownsampler(nn.Module):
#     """
#     Downsample trajectory from high-frequency (T_origin) to match 3D VAE latent temporal resolution (T_latent).
    
#     Input: [B*Nc*T_origin, L_traj, C_in]
#     Output: [B*Nc*T_latent, L_traj, hidden_size]
#     """
#     def __init__(self, in_channels: int, out_channels: int, compression_rate: int = 4, T_origin: int = 16):
#         super().__init__()
#         self.compression_rate = compression_rate
#         self.T_origin = T_origin
#         self.T_latent = T_origin // compression_rate
#         self.in_channels = in_channels
#         self.out_channels = out_channels
        
#         if self.T_origin % self.compression_rate != 0:
#             raise ValueError(
#                 f"T_origin ({T_origin}) must be divisible by compression_rate ({compression_rate})"
#             )
        
#         # Temporal pooling
#         if compression_rate > 1:
#             self.temporal_pool = nn.AvgPool1d(
#                 kernel_size=compression_rate,
#                 stride=compression_rate
#             )
#         else:
#             self.temporal_pool = None
        
#         # Project to hidden dimension
#         self.proj = nn.Sequential(
#             nn.Linear(in_channels, out_channels),
#             nn.LayerNorm(out_channels),
#             nn.GELU(),
#             nn.Linear(out_channels, out_channels),
#         )
    
#     def forward(self, traj: Tensor) -> Tensor:
#         """
#         Args:
#             traj: [B*Nc*T_origin, L_traj, C_in] where C_in = 3 (xyz)
#         Returns:
#             traj_down: [B*Nc*T_latent, L_traj, hidden_size]
#         """
#         B_full_origin, L_traj, C_in = traj.shape
        
#         if self.temporal_pool is None:
#             # No temporal downsampling needed
#             return self.proj(traj)
        
#         # Validate input shape
#         if B_full_origin % self.T_origin != 0:
#             raise ValueError(
#                 f"Input batch size ({B_full_origin}) must be divisible by T_origin ({self.T_origin}). "
#                 f"Expected shape: [B*Nc*T_origin, L_traj, C_in] = [B*Nc*{self.T_origin}, {L_traj}, {C_in}]"
#             )
        
#         # Calculate B*Nc
#         B_Nc = B_full_origin // self.T_origin
        
#         # Reshape to separate temporal dimension
#         traj = traj.reshape(B_Nc, self.T_origin, L_traj, C_in)
        
#         # Rearrange for temporal pooling
#         traj = rearrange(traj, 'b t l c -> (b l c) t')
        
#         # Apply temporal pooling
#         traj = self.temporal_pool(traj)
        
#         # Reshape back
#         traj = rearrange(traj, '(b l c) t -> b t l c', b=B_Nc, l=L_traj, c=C_in)
        
#         # Flatten batch and time
#         B_full_latent = B_Nc * self.T_latent
#         traj = traj.reshape(B_full_latent, L_traj, C_in)
        
#         # Project to hidden dimension
#         traj = self.proj(traj)  # [B*Nc*T_latent, L_traj, hidden_size]
        
#         return traj


# class PhysicalDiT(nn.Module):
#     """
#     Physical DiT: Diffusion Transformer for physical world trajectory prediction.
    
#     Architecture: DoubleStream (traj & cond interaction) -> SingleStream (joint processing)
#     """
    
#     def __init__(self, params: PhysicalDiTParams):
#         super().__init__()
        
#         self.params = params
#         self.hidden_size = params.hidden_size
#         self.num_heads = params.num_heads
#         self.num_output_frames = params.num_output_frames
        
#         # Validate configuration
#         if params.hidden_size % params.num_heads != 0:
#             raise ValueError(
#                 f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
#             )
        
#         # Positional encoding
#         pe_dim = params.hidden_size // params.num_heads
#         if sum(params.axes_dim) != pe_dim:
#             raise ValueError(f"Got sum(axes_dim)={sum(params.axes_dim)} but expected positional dim {pe_dim}")
        
#         self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        
#         # Trajectory temporal downsampler
#         self.traj_downsampler = TrajectoryTemporalDownsampler(
#             in_channels=params.traj_in_channels,
#             out_channels=params.hidden_size,
#             compression_rate=params.traj_temporal_compression,
#             T_origin=params.T_origin
#         )
        
#         # Input projections
#         self.traj_in = nn.Linear(params.hidden_size, params.hidden_size, bias=True)
#         self.cond_in = nn.Linear(params.hidden_size, params.hidden_size, bias=True)
        
#         # Time and guidance embeddings
#         self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
#         self.guidance_in = (
#             MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) 
#             if params.guidance_embed else nn.Identity()
#         )
        
#         # Transformer blocks
#         self.double_blocks = nn.ModuleList([
#             DoubleStreamBlock(
#                 hidden_size=self.hidden_size,
#                 num_heads=self.num_heads,
#                 mlp_ratio=params.mlp_ratio,
#                 qkv_bias=params.qkv_bias,
#             )
#             for _ in range(params.depth)
#         ])
        
#         self.single_blocks = nn.ModuleList([
#             SingleStreamBlock(
#                 hidden_size=self.hidden_size,
#                 num_heads=self.num_heads,
#                 mlp_ratio=params.mlp_ratio
#             )
#             for _ in range(params.depth_single_blocks)
#         ])
        
#         # Final prediction layer
#         self.final_layer = LastLayer(
#             hidden_size=self.hidden_size,
#             patch_size=1,
#             out_channels=params.num_output_frames * params.traj_out_channels
#         )
    
#     def forward(
#         self,
#         traj: Tensor,  # [B*Nc*T_origin, L_traj, C_in=3]
#         traj_ids: Tensor,  # [B*Nc*T_latent, L_traj, 1] - temporal position ids
#         cond: Tensor,  # [B*Nc*T_latent, L_cond, hidden_size]
#         cond_ids: Tensor,  # [B*Nc*T_latent, L_cond, 1]
#         timesteps: Tensor,  # [B*Nc*T_latent]
#         guidance: Tensor | None = None,  # [B*Nc*T_latent]
#     ) -> Tensor:
#         """
#         Forward pass of Physical DiT.
        
#         Returns:
#             pred: Predicted trajectory [B*Nc*T_latent, L_traj, num_frames*3]
#         """
#         # Validate input dimensions
#         if traj.ndim != 3:
#             raise ValueError(f"Expected traj to be 3D [B, L, C], got shape {traj.shape}")
#         if cond.ndim != 3:
#             raise ValueError(f"Expected cond to be 3D [B, L, C], got shape {cond.shape}")
        
#         # Step 1: Downsample and project trajectory
#         traj = self.traj_downsampler(traj)  # [B*Nc*T_origin, L, 3] -> [B*Nc*T_latent, L, hidden]
        
#         # CRITICAL: Ensure traj is 3D after downsampling
#         if traj.ndim != 3:
#             raise ValueError(f"After downsampling, traj should be 3D, got shape {traj.shape}")
        
#         traj = self.traj_in(traj)  # [B*Nc*T_latent, L, hidden]
        
#         # CRITICAL: Validate traj shape after projection
#         if traj.ndim != 3:
#             raise ValueError(f"After traj_in, traj should be 3D, got shape {traj.shape}")
        
#         # Step 2: Project condition
#         cond = self.cond_in(cond)  # [B*Nc*T_latent, L, hidden]
        
#         if cond.ndim != 3:
#             raise ValueError(f"After cond_in, cond should be 3D, got shape {cond.shape}")
        
#         # Step 3: Time and guidance embeddings
#         vec = self.time_in(timestep_embedding(timesteps, 256))  # [B*Nc*T_latent, hidden]
        
#         if self.params.guidance_embed:
#             if guidance is None:
#                 raise ValueError("Guidance required for guidance-distilled model")
#             vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        
#         # Step 4: Generate position embeddings
#         # Concatenate position IDs: [B, L_cond+L_traj, 1]
#         ids = torch.cat([cond_ids, traj_ids], dim=1)
#         pe = self.pe_embedder(ids)  # [B, 1, L_cond+L_traj, pe_dim//2, 2, 2]
        
#         # Step 5: DoubleStream blocks
#         for block in self.double_blocks:
#             traj, cond = block(img=traj, cond=cond, vec=vec, pe=pe)
            
#             # Validate shapes remain 3D
#             if traj.ndim != 3 or cond.ndim != 3:
#                 raise ValueError(
#                     f"After DoubleStreamBlock, expected 3D tensors, "
#                     f"got traj.shape={traj.shape}, cond.shape={cond.shape}"
#                 )
        
#         # Step 6: Concatenate streams
#         x = torch.cat([cond, traj], dim=1)  # [B, L_cond+L_traj, hidden]
        
#         # Step 7: SingleStream blocks
#         for block in self.single_blocks:
#             x = block(x, vec=vec, pe=pe)
            
#             if x.ndim != 3:
#                 raise ValueError(f"After SingleStreamBlock, expected 3D tensor, got shape {x.shape}")
        
#         # Step 8: Extract trajectory tokens and predict
#         traj_out = x[:, cond.shape[1]:, ...]  # [B, L_traj, hidden]
#         traj_out = self.final_layer(traj_out, vec)  # [B, L_traj, num_frames*3]
        
#         return traj_out
    
#     def training_losses(
#         self,
#         traj: Tensor,
#         traj_ids: Tensor,
#         cond: Tensor,
#         cond_ids: Tensor,
#         t: Tensor,
#         target_traj: Tensor,
#         guidance: Tensor | None = None,
#         noise: Tensor | None = None,
#         return_predict: bool = False
#     ) -> dict:
#         """Compute training losses for diffusion model."""
#         if noise is None:
#             noise = torch.randn_like(target_traj)
        
#         terms = {}
        
#         # Diffusion forward process
#         t_reshaped = t[:, None, None]
#         x_t = t_reshaped * target_traj + (1.0 - t_reshaped) * noise
        
#         # Predict
#         pred = self.forward(
#             traj=traj,
#             traj_ids=traj_ids,
#             cond=cond,
#             cond_ids=cond_ids,
#             timesteps=t,
#             guidance=guidance
#         )
        
#         # Velocity prediction target
#         target = target_traj - noise
        
#         # MSE loss
#         assert pred.shape == target.shape == target_traj.shape
#         terms["mse"] = mean_flat((target - pred) ** 2)
#         terms["loss"] = terms["mse"].mean()
        
#         if return_predict:
#             predict = x_t + pred * (1.0 - t_reshaped)
#             terms["predict"] = predict
#         else:
#             terms["predict"] = None
        
#         return terms
    
#     @torch.no_grad()
#     def generate(
#         self,
#         cond: Tensor,
#         cond_ids: Tensor,
#         L_traj: int,
#         num_sampling_steps: int = 50,
#         guidance_scale: float = 1.0,
#         device: torch.device = None,
#     ) -> Tensor:
#         """Generate trajectory from scratch."""
#         if device is None:
#             device = cond.device
        
#         B_latent = cond.shape[0]
#         num_axes = cond_ids.shape[-1]
        
#         # Create trajectory position IDs
#         traj_ids = torch.arange(L_traj, device=device)[None, :, None].expand(B_latent, -1, num_axes)
        
#         # Start from pure noise
#         x_t = torch.randn(
#             B_latent, L_traj, self.num_output_frames * self.params.traj_out_channels,
#             device=device, dtype=cond.dtype
#         )
        
#         # Sampling schedule
#         timesteps = torch.linspace(1.0, 0.0, num_sampling_steps + 1, device=device).tolist()
        
#         # Guidance
#         guidance = None
#         if self.params.guidance_embed and guidance_scale != 1.0:
#             guidance = torch.full((B_latent,), guidance_scale, device=device, dtype=cond.dtype)
        
#         # Dummy trajectory input
#         T_latent = self.params.T_origin // self.params.traj_temporal_compression
#         B_Nc = B_latent // T_latent
#         traj_dummy = torch.zeros(
#             B_Nc * self.params.T_origin, L_traj, self.params.traj_in_channels,
#             device=device, dtype=cond.dtype
#         )
        
#         # Sampling loop
#         for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
#             t_vec = torch.full((B_latent,), t_curr, dtype=cond.dtype, device=device)
            
#             pred = self.forward(
#                 traj=traj_dummy,
#                 traj_ids=traj_ids,
#                 cond=cond,
#                 cond_ids=cond_ids,
#                 timesteps=t_vec,
#                 guidance=guidance
#             )
            
#             x_t = x_t + (t_prev - t_curr) * pred
        
#         return x_t


# # ==========================================
# # Testing
# # ==========================================
# if __name__ == "__main__":
#     print("=" * 80)
#     print("Testing Physical DiT")
#     print("=" * 80)
    
#     # Configuration
#     B = 2
#     Nc = 6
#     T_latent = 4
#     T_origin = T_latent * 4
    
#     L_img = 256
#     L_pose = 4
#     L_cond = L_img + L_pose
#     L_traj = 12
    
#     hidden_size = 1024
#     num_output_frames = 12
    
#     # Create model
#     params = PhysicalDiTParams(
#         traj_in_channels=3,
#         traj_out_channels=3,
#         hidden_size=hidden_size,
#         num_heads=16,
#         depth=1,
#         depth_single_blocks=1,
#         traj_temporal_compression=4,
#         T_origin=T_origin,
#         num_output_frames=num_output_frames,
#         guidance_embed=True,
#     )
    
#     model = PhysicalDiT(params).cuda()
    
#     print(f"\nModel Configuration:")
#     print(f"  Hidden size: {params.hidden_size}")
#     print(f"  Num heads: {params.num_heads}")
#     print(f"  PE dim: {params.hidden_size // params.num_heads}")
#     print(f"  axes_dim: {params.axes_dim}")
    
#     # Test 1: Forward pass
#     print("\n" + "=" * 80)
#     print("Test 1: Forward Pass")
#     print("=" * 80)
    
#     B_full_latent = B * Nc * T_latent
#     B_full_origin = B * Nc * T_origin
    
#     # Create inputs with CORRECT shapes
#     traj_input = torch.randn(B_full_origin, L_traj, 3).cuda()
#     traj_ids = torch.arange(L_traj, device='cuda')[None, :, None].expand(B_full_latent, -1, 1)
    
#     cond_input = torch.randn(B_full_latent, L_cond, hidden_size).cuda()
#     cond_ids = torch.arange(L_cond, device='cuda')[None, :, None].expand(B_full_latent, -1, 1)
    
#     timesteps = torch.rand(B_full_latent).cuda()
#     guidance = torch.ones(B_full_latent).cuda() * 1.5
    
#     print(f"Input shapes:")
#     print(f"  traj_input: {traj_input.shape} - MUST be 3D")
#     print(f"  traj_ids: {traj_ids.shape}")
#     print(f"  cond_input: {cond_input.shape} - MUST be 3D")
#     print(f"  cond_ids: {cond_ids.shape}")
    
#     # Forward
#     try:
#         output = model(
#             traj=traj_input,
#             traj_ids=traj_ids,
#             cond=cond_input,
#             cond_ids=cond_ids,
#             timesteps=timesteps,
#             guidance=guidance
#         )
        
#         expected_shape = (B_full_latent, L_traj, num_output_frames * 3)
#         print(f"\nOutput shape: {output.shape}")
#         print(f"Expected shape: {expected_shape}")
#         assert output.shape == expected_shape
#         print("✅ Forward pass successful!")
        
#     except Exception as e:
#         print(f"❌ Forward pass failed with error:")
#         print(f"   {type(e).__name__}: {e}")
#         import traceback
#         traceback.print_exc()
#         raise
    
#     # Test 2: Training losses
#     print("\n" + "=" * 80)
#     print("Test 2: Training Losses")
#     print("=" * 80)
    
#     target_traj = torch.randn(B_full_latent, L_traj, num_output_frames * 3).cuda()
#     t = torch.rand(B_full_latent).cuda()
    
#     losses = model.training_losses(
#         traj=traj_input,
#         traj_ids=traj_ids,
#         cond=cond_input,
#         cond_ids=cond_ids,
#         t=t,
#         target_traj=target_traj,
#         guidance=guidance,
#         return_predict=True
#     )
    
#     print(f"Loss: {losses['loss'].item():.6f}")
#     print("✅ Training losses computed successfully!")
    
#     # Test 3: Generation
#     print("\n" + "=" * 80)
#     print("Test 3: Generation")
#     print("=" * 80)
    
#     generated = model.generate(
#         cond=cond_input,
#         cond_ids=cond_ids,
#         L_traj=L_traj,
#         num_sampling_steps=20,
#         guidance_scale=1.5,
#     )
    
#     print(f"Generated shape: {generated.shape}")
#     print("✅ Generation successful!")
    
#     print("\n" + "=" * 80)
#     print("🎉 All Tests Passed!")
#     print("=" * 80)
    
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"\nTotal parameters: {total_params:,}")
#     print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from einops import rearrange
import math

# Import the DiT modules from your existing codebase
from models.modules.dit_modules.layers import (
    DoubleStreamBlock,
    SingleStreamBlock,
    timestep_embedding,
    MLPEmbedder,
    EmbedND,
    LastLayer
)


def mean_flat(tensor):
    """Average tensor over all dimensions except batch."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


@dataclass
class PhysicalDiTParams:
    """Configuration for Physical DiT model."""
    # Trajectory input/output configuration
    traj_in_channels: int = 3  # xyz coordinates
    traj_out_channels: int = 3  # predict xyz
    
    # Model architecture
    hidden_size: int = 1024
    mlp_ratio: float = 4.0
    num_heads: int = 16
    depth: int = 8  # Number of DoubleStream blocks
    depth_single_blocks: int = 16  # Number of SingleStream blocks
    
    # Trajectory temporal downsampling
    traj_temporal_compression: int = 4  # Match 3D VAE compression
    T_origin: int = 16  # Original temporal length before compression
    num_output_frames: int = 12  # Predict 12 single frames
    
    # RoPE configuration
    axes_dim: list[int] = None  # Will be set based on hidden_size
    theta: int = 10000
    qkv_bias: bool = True
    
    # Guidance
    guidance_embed: bool = True
    
    def __post_init__(self):
        if self.axes_dim is None:
            # pe_dim must equal hidden_size // num_heads
            pe_dim = self.hidden_size // self.num_heads
            # For single temporal axis
            self.axes_dim = [pe_dim]
        
        # Validate temporal compression
        if self.T_origin % self.traj_temporal_compression != 0:
            raise ValueError(
                f"T_origin ({self.T_origin}) must be divisible by "
                f"traj_temporal_compression ({self.traj_temporal_compression})"
            )


class TrajectoryTemporalDownsampler(nn.Module):
    """
    Downsample trajectory from high-frequency (T_origin) to match 3D VAE latent temporal resolution (T_latent).
    
    Input: [B*Nc*T_origin, L_traj, C_in]
    Output: [B*Nc*T_latent, L_traj, hidden_size]
    """
    def __init__(self, in_channels: int, out_channels: int, compression_rate: int = 4, T_origin: int = 16):
        super().__init__()
        self.compression_rate = compression_rate
        self.T_origin = T_origin
        self.T_latent = T_origin // compression_rate
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if self.T_origin % self.compression_rate != 0:
            raise ValueError(
                f"T_origin ({T_origin}) must be divisible by compression_rate ({compression_rate})"
            )
        
        # Temporal pooling
        if compression_rate > 1:
            self.temporal_pool = nn.AvgPool1d(
                kernel_size=compression_rate,
                stride=compression_rate
            )
        else:
            self.temporal_pool = None
        
        # Project to hidden dimension
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
    
    def forward(self, traj: Tensor) -> Tensor:
        """
        Args:
            traj: [B*Nc*T_origin, L_traj, C_in] where C_in = 3 (xyz)
        Returns:
            traj_down: [B*Nc*T_latent, L_traj, hidden_size]
        """
        B_full_origin, L_traj, C_in = traj.shape
        
        if self.temporal_pool is None:
            # No temporal downsampling needed
            return self.proj(traj)
        
        # Validate input shape
        if B_full_origin % self.T_origin != 0:
            raise ValueError(
                f"Input batch size ({B_full_origin}) must be divisible by T_origin ({self.T_origin}). "
                f"Expected shape: [B*Nc*T_origin, L_traj, C_in] = [B*Nc*{self.T_origin}, {L_traj}, {C_in}]"
            )
        
        # Calculate B*Nc
        B_Nc = B_full_origin // self.T_origin
        
        # Reshape to separate temporal dimension
        traj = traj.reshape(B_Nc, self.T_origin, L_traj, C_in)
        
        # Rearrange for temporal pooling
        traj = rearrange(traj, 'b t l c -> (b l c) t')
        
        # Apply temporal pooling
        traj = self.temporal_pool(traj)
        
        # Reshape back
        traj = rearrange(traj, '(b l c) t -> b t l c', b=B_Nc, l=L_traj, c=C_in)
        
        # Flatten batch and time
        B_full_latent = B_Nc * self.T_latent
        traj = traj.reshape(B_full_latent, L_traj, C_in)
        
        # Project to hidden dimension
        traj = self.proj(traj)  # [B*Nc*T_latent, L_traj, hidden_size]
        
        # CRITICAL: Ensure output is exactly 3D
        assert traj.ndim == 3, f"Downsampler output must be 3D, got {traj.shape}"
        assert traj.shape == (B_full_latent, L_traj, self.out_channels), \
            f"Expected shape [{B_full_latent}, {L_traj}, {self.out_channels}], got {traj.shape}"
        
        return traj


class PhysicalDiT(nn.Module):
    """
    Physical DiT: Diffusion Transformer for physical world trajectory prediction.
    
    Architecture: DoubleStream (traj & cond interaction) -> SingleStream (joint processing)
    """
    
    def __init__(self, params: PhysicalDiTParams):
        super().__init__()
        
        self.params = params
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.num_output_frames = params.num_output_frames
        
        # Validate configuration
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        
        # Positional encoding
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got sum(axes_dim)={sum(params.axes_dim)} but expected positional dim {pe_dim}")
        
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        
        # Trajectory temporal downsampler
        self.traj_downsampler = TrajectoryTemporalDownsampler(
            in_channels=params.traj_in_channels,
            out_channels=params.hidden_size,
            compression_rate=params.traj_temporal_compression,
            T_origin=params.T_origin
        )
        
        # Input projections
        self.traj_in = nn.Linear(params.hidden_size, params.hidden_size, bias=True)
        self.cond_in = nn.Linear(params.hidden_size, params.hidden_size, bias=True)
        
        # Time and guidance embeddings
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) 
            if params.guidance_embed else nn.Identity()
        )
        
        # Transformer blocks
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=params.mlp_ratio,
                qkv_bias=params.qkv_bias,
            )
            for _ in range(params.depth)
        ])
        
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=params.mlp_ratio
            )
            for _ in range(params.depth_single_blocks)
        ])
        
        # Final prediction layer
        self.final_layer = LastLayer(
            hidden_size=self.hidden_size,
            patch_size=1,
            out_channels=params.num_output_frames * params.traj_out_channels
        )
    
    def forward(
        self,
        traj: Tensor,  # [B*Nc*T_origin, L_traj, C_in=3]
        traj_ids: Tensor,  # [B*Nc*T_latent, L_traj, 1] - temporal position ids
        cond: Tensor,  # [B*Nc*T_latent, L_cond, hidden_size]
        cond_ids: Tensor,  # [B*Nc*T_latent, L_cond, 1]
        timesteps: Tensor,  # [B*Nc*T_latent]
        guidance: Tensor | None = None,  # [B*Nc*T_latent]
    ) -> Tensor:
        """
        Forward pass of Physical DiT.
        
        Returns:
            pred: Predicted trajectory [B*Nc*T_latent, L_traj, num_frames*3]
        """
        # Validate input dimensions
        assert traj.ndim == 3, f"traj must be 3D, got {traj.shape}"
        assert cond.ndim == 3, f"cond must be 3D, got {cond.shape}"
        assert timesteps.ndim == 1, f"timesteps must be 1D, got {timesteps.shape}"
        
        B_latent = cond.shape[0]
        L_cond = cond.shape[1]
        L_traj = traj_ids.shape[1]
        
        print(f"\n=== PhysicalDiT Forward ===")
        print(f"Input traj: {traj.shape}")
        print(f"Input cond: {cond.shape}")
        print(f"timesteps: {timesteps.shape}")
        
        # Step 1: Downsample and project trajectory
        traj = self.traj_downsampler(traj)
        print(f"After downsampler: {traj.shape}")
        assert traj.shape == (B_latent, L_traj, self.hidden_size), \
            f"After downsampler, expected [{B_latent}, {L_traj}, {self.hidden_size}], got {traj.shape}"
        
        traj = self.traj_in(traj)
        print(f"After traj_in: {traj.shape}")
        assert traj.shape == (B_latent, L_traj, self.hidden_size), \
            f"After traj_in, expected [{B_latent}, {L_traj}, {self.hidden_size}], got {traj.shape}"
        
        # Step 2: Project condition
        cond = self.cond_in(cond)
        print(f"After cond_in: {cond.shape}")
        assert cond.shape == (B_latent, L_cond, self.hidden_size), \
            f"After cond_in, expected [{B_latent}, {L_cond}, {self.hidden_size}], got {cond.shape}"
        
        # Step 3: Time and guidance embeddings
        vec = self.time_in(timestep_embedding(timesteps, 256))
        print(f"vec shape: {vec.shape}")
        assert vec.shape == (B_latent, self.hidden_size), \
            f"vec should be [{B_latent}, {self.hidden_size}], got {vec.shape}"
        
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Guidance required for guidance-distilled model")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
            print(f"vec after guidance: {vec.shape}")
        
        # Step 4: Generate position embeddings
        ids = torch.cat([cond_ids, traj_ids], dim=1)
        print(f"ids shape: {ids.shape}")
        pe = self.pe_embedder(ids)
        print(f"pe shape: {pe.shape}")
        
        # Step 5: DoubleStream blocks
        for i, block in enumerate(self.double_blocks):
            print(f"\n--- DoubleStreamBlock {i} ---")
            print(f"  Input traj: {traj.shape} (MUST be 3D)")
            print(f"  Input cond: {cond.shape} (MUST be 3D)")
            print(f"  Input vec: {vec.shape} (MUST be 2D)")
            print(f"  Input pe: {pe.shape}")
            
            # CRITICAL: Verify shapes before passing to block
            assert traj.ndim == 3, f"traj must be 3D before block, got {traj.shape}"
            assert cond.ndim == 3, f"cond must be 3D before block, got {cond.shape}"
            assert vec.ndim == 2, f"vec must be 2D before block, got {vec.shape}"
            
            traj, cond = block(img=traj, cond=cond, vec=vec, pe=pe)
            
            print(f"  Output traj: {traj.shape}")
            print(f"  Output cond: {cond.shape}")
            
            assert traj.ndim == 3, f"traj must remain 3D after block, got {traj.shape}"
            assert cond.ndim == 3, f"cond must remain 3D after block, got {cond.shape}"
        
        # Step 6: Concatenate streams
        x = torch.cat([cond, traj], dim=1)
        print(f"\nAfter concat: {x.shape}")
        
        # Step 7: SingleStream blocks
        for i, block in enumerate(self.single_blocks):
            x = block(x, vec=vec, pe=pe)
            assert x.ndim == 3, f"x must remain 3D after SingleStreamBlock, got {x.shape}"
        
        # Step 8: Extract trajectory tokens and predict
        traj_out = x[:, L_cond:, ...]
        print(f"traj_out before final_layer: {traj_out.shape}")
        traj_out = self.final_layer(traj_out, vec)
        print(f"Final output: {traj_out.shape}")
        
        return traj_out
    
    def training_losses(
        self,
        traj: Tensor,
        traj_ids: Tensor,
        cond: Tensor,
        cond_ids: Tensor,
        t: Tensor,
        target_traj: Tensor,
        guidance: Tensor | None = None,
        noise: Tensor | None = None,
        return_predict: bool = False
    ) -> dict:
        """Compute training losses for diffusion model."""
        if noise is None:
            noise = torch.randn_like(target_traj)
        
        terms = {}
        
        # Diffusion forward process
        t_reshaped = t[:, None, None]
        x_t = t_reshaped * target_traj + (1.0 - t_reshaped) * noise
        
        # Predict
        pred = self.forward(
            traj=traj,
            traj_ids=traj_ids,
            cond=cond,
            cond_ids=cond_ids,
            timesteps=t,
            guidance=guidance
        )
        
        # Velocity prediction target
        target = target_traj - noise
        
        # MSE loss
        assert pred.shape == target.shape == target_traj.shape
        terms["mse"] = mean_flat((target - pred) ** 2)
        terms["loss"] = terms["mse"].mean()
        
        if return_predict:
            predict = x_t + pred * (1.0 - t_reshaped)
            terms["predict"] = predict
        else:
            terms["predict"] = None
        
        return terms
    
    @torch.no_grad()
    def generate(
        self,
        cond: Tensor,
        cond_ids: Tensor,
        L_traj: int,
        num_sampling_steps: int = 50,
        guidance_scale: float = 1.0,
        device: torch.device = None,
    ) -> Tensor:
        """Generate trajectory from scratch."""
        if device is None:
            device = cond.device
        
        B_latent = cond.shape[0]
        num_axes = cond_ids.shape[-1]
        
        # Create trajectory position IDs
        traj_ids = torch.arange(L_traj, device=device)[None, :, None].expand(B_latent, -1, num_axes)
        
        # Start from pure noise
        x_t = torch.randn(
            B_latent, L_traj, self.num_output_frames * self.params.traj_out_channels,
            device=device, dtype=cond.dtype
        )
        
        # Sampling schedule
        timesteps = torch.linspace(1.0, 0.0, num_sampling_steps + 1, device=device).tolist()
        
        # Guidance
        guidance = None
        if self.params.guidance_embed and guidance_scale != 1.0:
            guidance = torch.full((B_latent,), guidance_scale, device=device, dtype=cond.dtype)
        
        # Dummy trajectory input
        T_latent = self.params.T_origin // self.params.traj_temporal_compression
        B_Nc = B_latent // T_latent
        traj_dummy = torch.zeros(
            B_Nc * self.params.T_origin, L_traj, self.params.traj_in_channels,
            device=device, dtype=cond.dtype
        )
        
        # Sampling loop
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((B_latent,), t_curr, dtype=cond.dtype, device=device)
            
            pred = self.forward(
                traj=traj_dummy,
                traj_ids=traj_ids,
                cond=cond,
                cond_ids=cond_ids,
                timesteps=t_vec,
                guidance=guidance
            )
            
            x_t = x_t + (t_prev - t_curr) * pred
        
        return x_t


# ==========================================
# Testing
# ==========================================
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Physical DiT")
    print("=" * 80)
    
    # Configuration
    B = 2
    Nc = 6
    T_latent = 4
    T_origin = T_latent * 4
    
    L_img = 256
    L_pose = 4
    L_cond = L_img + L_pose
    L_traj = 12
    
    hidden_size = 1024
    num_output_frames = 12
    
    # Create model
    params = PhysicalDiTParams(
        traj_in_channels=3,
        traj_out_channels=3,
        hidden_size=hidden_size,
        num_heads=16,
        depth=1,
        depth_single_blocks=1,
        traj_temporal_compression=4,
        T_origin=T_origin,
        num_output_frames=num_output_frames,
        guidance_embed=True,
    )
    
    model = PhysicalDiT(params).cuda()
    
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {params.hidden_size}")
    print(f"  Num heads: {params.num_heads}")
    print(f"  PE dim: {params.hidden_size // params.num_heads}")
    print(f"  axes_dim: {params.axes_dim}")
    
    # Test 1: Forward pass
    print("\n" + "=" * 80)
    print("Test 1: Forward Pass")
    print("=" * 80)
    
    B_full_latent = B * Nc * T_latent
    B_full_origin = B * Nc * T_origin
    
    # Create inputs with CORRECT shapes
    traj_input = torch.randn(B_full_origin, L_traj, 3).cuda()
    traj_ids = torch.arange(L_traj, device='cuda')[None, :, None].expand(B_full_latent, -1, 1)
    
    cond_input = torch.randn(B_full_latent, L_cond, hidden_size).cuda()
    cond_ids = torch.arange(L_cond, device='cuda')[None, :, None].expand(B_full_latent, -1, 1)
    
    timesteps = torch.rand(B_full_latent).cuda()
    guidance = torch.ones(B_full_latent).cuda() * 1.5
    
    print(f"Input shapes:")
    print(f"  traj_input: {traj_input.shape} - MUST be 3D")
    print(f"  traj_ids: {traj_ids.shape}")
    print(f"  cond_input: {cond_input.shape} - MUST be 3D")
    print(f"  cond_ids: {cond_ids.shape}")
    
    # Forward
    try:
        output = model(
            traj=traj_input,
            traj_ids=traj_ids,
            cond=cond_input,
            cond_ids=cond_ids,
            timesteps=timesteps,
            guidance=guidance
        )
        
        expected_shape = (B_full_latent, L_traj, num_output_frames * 3)
        print(f"\n✅ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: {expected_shape}")
        assert output.shape == expected_shape
        
    except Exception as e:
        print(f"❌ Forward pass failed:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n" + "=" * 80)
    print("🎉 Test Passed!")
    print("=" * 80)