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
#     traj_in_channels: int = 3  # xyz coordinates (not xy like Epona)
#     traj_out_channels: int = 3  # predict xyz
    
#     # Model architecture
#     hidden_size: int = 1024
#     mlp_ratio: float = 4.0
#     num_heads: int = 16
#     depth: int = 8  # Number of DoubleStream blocks
#     depth_single_blocks: int = 16  # Number of SingleStream blocks
    
#     # Trajectory temporal downsampling
#     traj_temporal_compression: int = 4  # Match 3D VAE compression (every 4 frames)
#     T_origin: int = 16  # Original temporal length before compression
#     num_output_frames: int = 12  # Predict 12 single frames
    
#     # RoPE configuration
#     axes_dim: list[int] = None  # [time_dim, ...], will be set based on hidden_size
#     theta: int = 10000
#     qkv_bias: bool = True
    
#     # Guidance
#     guidance_embed: bool = True
    
#     def __post_init__(self):
#         if self.axes_dim is None:
#             # Default: allocate positional encoding dimensions
#             pe_dim = self.hidden_size // self.num_heads
#             # For trajectory, we mainly care about temporal dimension
#             self.axes_dim = [pe_dim]  # Single axis for time
        
#         # Validate temporal compression
#         if self.T_origin % self.traj_temporal_compression != 0:
#             raise ValueError(
#                 f"T_origin ({self.T_origin}) must be divisible by "
#                 f"traj_temporal_compression ({self.traj_temporal_compression})"
#             )


# class TrajectoryTemporalDownsampler(nn.Module):
#     """
#     Downsample trajectory from high-frequency (T_origin) to match 3D VAE latent temporal resolution (T_latent).
    
#     Key insight: The temporal dimension is embedded in the batch dimension.
#     Input: [B*Nc*T_origin, L_traj, C_in]
#     Output: [B*Nc*T_latent, L_traj, hidden_size]
    
#     where T_latent = T_origin // compression_rate
    
#     Process: 
#     1. Reshape: [B*Nc*T_origin, L, C] -> [B*Nc, T_origin, L, C]
#     2. Pool: [B*Nc, T_origin, L, C] -> [B*Nc, T_latent, L, C]
#     3. Flatten: [B*Nc, T_latent, L, C] -> [B*Nc*T_latent, L, C]
#     4. Project to hidden_size
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
        
#         # Step 1: Reshape to separate temporal dimension
#         # [B*Nc*T_origin, L_traj, C_in] -> [B*Nc, T_origin, L_traj, C_in]
#         traj = traj.reshape(B_Nc, self.T_origin, L_traj, C_in)
        
#         # Step 2: Rearrange for temporal pooling
#         # [B*Nc, T_origin, L_traj, C_in] -> [B*Nc*L_traj*C_in, T_origin]
#         traj = rearrange(traj, 'b t l c -> (b l c) t')
        
#         # Step 3: Apply temporal pooling
#         # [B*Nc*L_traj*C_in, T_origin] -> [B*Nc*L_traj*C_in, T_latent]
#         traj = self.temporal_pool(traj)
        
#         # Step 4: Reshape back
#         # [B*Nc*L_traj*C_in, T_latent] -> [B*Nc, T_latent, L_traj, C_in]
#         traj = rearrange(traj, '(b l c) t -> b t l c', b=B_Nc, l=L_traj, c=C_in)
        
#         # Step 5: Flatten batch and time
#         # [B*Nc, T_latent, L_traj, C_in] -> [B*Nc*T_latent, L_traj, C_in]
#         B_full_latent = B_Nc * self.T_latent
#         traj = traj.reshape(B_full_latent, L_traj, C_in)
        
#         # Step 6: Project to hidden dimension
#         traj = self.proj(traj)  # [B*Nc*T_latent, L_traj, hidden_size]
        
#         return traj


# class PhysicalDiT(nn.Module):
#     """
#     Physical DiT: Diffusion Transformer for physical world trajectory prediction.
    
#     Architecture: DoubleStream (traj & cond interaction) -> SingleStream (joint processing)
    
#     Key differences from Epona's Traj DiT:
#     1. Predicts xyz trajectories (3D) instead of xy (2D)
#     2. Uses 3D VAE latent features as condition
#     3. Predicts 12 single frames instead of continuous trajectory
#     4. Includes trajectory temporal downsampling to match 3D VAE compression
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
#             raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        
#         self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        
#         # Trajectory temporal downsampler and projection to hidden_size
#         self.traj_downsampler = TrajectoryTemporalDownsampler(
#             in_channels=params.traj_in_channels,
#             out_channels=params.hidden_size,
#             compression_rate=params.traj_temporal_compression,
#             T_origin=params.T_origin
#         )
        
#         # Trajectory input projection (after downsampling, project to hidden_size)
#         self.traj_in = nn.Linear(params.hidden_size, params.hidden_size, bias=True)
        
#         # Condition input projection (from mvstt output)
#         # mvstt outputs: (B*Nc*T, L_img+L_pose, hidden_size)
#         self.cond_in = nn.Linear(params.hidden_size, params.hidden_size, bias=True)
        
#         # Time embedding
#         self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        
#         # Guidance embedding (for classifier-free guidance)
#         self.guidance_in = (
#             MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) 
#             if params.guidance_embed else nn.Identity()
#         )
        
#         # Double-stream transformer blocks (trajectory and condition interaction)
#         self.double_blocks = nn.ModuleList([
#             DoubleStreamBlock(
#                 hidden_size=self.hidden_size,
#                 num_heads=self.num_heads,
#                 mlp_ratio=params.mlp_ratio,
#                 qkv_bias=params.qkv_bias,
#             )
#             for _ in range(params.depth)
#         ])
        
#         # Single-stream transformer blocks (after concatenating traj and cond)
#         self.single_blocks = nn.ModuleList([
#             SingleStreamBlock(
#                 hidden_size=self.hidden_size,
#                 num_heads=self.num_heads,
#                 mlp_ratio=params.mlp_ratio
#             )
#             for _ in range(params.depth_single_blocks)
#         ])
        
#         # Final prediction layer (LastLayer from Epona)
#         # Predict 12 frames * 3 channels (xyz) per trajectory token
#         self.final_layer = LastLayer(
#             hidden_size=self.hidden_size,
#             patch_size=1,  # We're not using patches for trajectory
#             out_channels=params.num_output_frames * params.traj_out_channels
#         )
    
#     def forward(
#         self,
#         traj: Tensor,  # [B*Nc*T_origin, L_traj, C_in=3]
#         traj_ids: Tensor,  # [B*Nc*T_latent, L_traj, 1] - temporal position ids
#         cond: Tensor,  # [B*Nc*T_latent, L_cond, hidden_size] - from mvstt
#         cond_ids: Tensor,  # [B*Nc*T_latent, L_cond, 1] - position ids for condition
#         timesteps: Tensor,  # [B*Nc*T_latent] - diffusion timesteps
#         guidance: Tensor | None = None,  # [B*Nc*T_latent] - guidance strength
#     ) -> Tensor:
#         """
#         Forward pass of Physical DiT.
        
#         Architecture flow:
#         1. Downsample trajectory and project both streams to hidden_size
#         2. Generate time and guidance embeddings
#         3. DoubleStreamBlocks: trajectory and condition interact
#         4. Concatenate the two streams
#         5. SingleStreamBlocks: process joint sequence
#         6. Extract trajectory tokens and predict
        
#         Args:
#             traj: Raw trajectory input [B*Nc*T_origin, L_traj, 3]
#             traj_ids: Position IDs for trajectory tokens
#             cond: Conditional features from mvstt [B*Nc*T_latent, L_cond, hidden_size]
#             cond_ids: Position IDs for conditional tokens
#             timesteps: Diffusion timesteps
#             guidance: Guidance strength for classifier-free guidance
            
#         Returns:
#             pred: Predicted trajectory [B*Nc*T_latent, L_traj, num_frames*3]
#         """
#         # Validate dimensions
#         if traj.ndim != 3 or cond.ndim != 3:
#             raise ValueError("Input traj and cond tensors must have 3 dimensions.")
        
#         # ==========================================
#         # Step 1: Project both streams to hidden_size
#         # ==========================================
#         # Downsample and project trajectory: [B*Nc*T_origin, L_traj, 3] -> [B*Nc*T_latent, L_traj, hidden_size]
#         traj = self.traj_downsampler(traj)
#         traj = self.traj_in(traj)  # Additional projection
        
#         # Project condition: [B*Nc*T_latent, L_cond, hidden_size] -> [B*Nc*T_latent, L_cond, hidden_size]
#         cond = self.cond_in(cond)
        
#         # ==========================================
#         # Step 2: Time and guidance embeddings
#         # ==========================================
#         vec = self.time_in(timestep_embedding(timesteps, 256))  # [B*Nc*T_latent, hidden_size]
        
#         # Add guidance if using classifier-free guidance
#         if self.params.guidance_embed:
#             if guidance is None:
#                 raise ValueError("Guidance strength required for guidance-distilled model")
#             vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        
#         # ==========================================
#         # Step 3: Prepare position embeddings
#         # ==========================================
#         # Generate position embeddings separately for cond and traj
#         # This is the KEY FIX: we need to handle PE correctly
#         cond_pe = self.pe_embedder(cond_ids)  # [B, 1, L_cond, pe_dim]
#         print(f"cond_ids.shape:{cond_ids.shape}")
#         print(f"cond_pe.shape:{cond_pe.shape}")
#         traj_pe = self.pe_embedder(traj_ids)  # [B, 1, L_traj, pe_dim]
#         print(f"traj_pe.shape:{traj_pe.shape}")
#         # Concatenate position embeddings for joint attention
#         pe = torch.cat([cond_pe, traj_pe], dim=2)  # [B, 1, L_cond+L_traj, pe_dim]
        
#         # ==========================================
#         # Step 4: DoubleStream blocks (traj & cond interaction)
#         # ==========================================
#         for block in self.double_blocks:
#             # img stream = traj, cond stream = cond
#             traj, cond = block(img=traj, cond=cond, vec=vec, pe=pe)
        
#         # ==========================================
#         # Step 5: Concatenate streams for single-stream processing
#         # ==========================================
#         # Concatenate: [cond_tokens; traj_tokens]
#         x = torch.cat([cond, traj], dim=1)  # [B*Nc*T_latent, L_cond+L_traj, hidden_size]
        
#         # ==========================================
#         # Step 6: SingleStream blocks (joint processing)
#         # ==========================================
#         for block in self.single_blocks:
#             x = block(x, vec=vec, pe=pe)
        
#         # ==========================================
#         # Step 7: Extract trajectory tokens and predict
#         # ==========================================
#         # Extract only trajectory tokens (skip condition tokens)
#         traj_out = x[:, cond.shape[1]:, ...]  # [B*Nc*T_latent, L_traj, hidden_size]
        
#         # Final prediction using LastLayer (from Epona)
#         traj_out = self.final_layer(traj_out, vec)  # [B*Nc*T_latent, L_traj, num_frames*3]
        
#         return traj_out
    
#     def training_losses(
#         self,
#         traj: Tensor,  # [B*Nc*T_origin, L_traj, C_in]
#         traj_ids: Tensor,
#         cond: Tensor,  # [B*Nc*T_latent, L_cond, hidden_size]
#         cond_ids: Tensor,
#         t: Tensor,  # [B*Nc*T_latent] - diffusion time
#         target_traj: Tensor,  # [B*Nc*T_latent, L_traj, num_frames*3] - ground truth
#         guidance: Tensor | None = None,
#         noise: Tensor | None = None,
#         return_predict: bool = False
#     ) -> dict:
#         """
#         Compute training losses for diffusion model.
        
#         Args:
#             traj: Input trajectory at diffusion time t
#             traj_ids: Position IDs for trajectory
#             cond: Conditional features from mvstt
#             cond_ids: Position IDs for condition
#             t: Diffusion timestep [0, 1]
#             target_traj: Ground truth trajectory (12 frames)
#             guidance: Guidance strength
#             noise: Noise to add (if None, sample randomly)
#             return_predict: Whether to return prediction
            
#         Returns:
#             Dictionary with losses and optional predictions
#         """
#         B_full, L_traj, C_in = traj.shape
#         target_shape = target_traj.shape
        
#         # Sample noise if not provided
#         if noise is None:
#             noise = torch.randn_like(target_traj)
        
#         terms = {}
        
#         # Diffusion forward process: x_t = t * x_0 + (1-t) * noise
#         # Reshape t for broadcasting: [B*Nc*T_latent] -> [B*Nc*T_latent, 1, 1]
#         t_reshaped = t[:, None, None]
#         x_t = t_reshaped * target_traj + (1.0 - t_reshaped) * noise
        
#         # We need to convert x_t back to trajectory format for input
#         # x_t is [B*Nc*T_latent, L_traj, num_frames*3]
#         # We need to convert it to [B*Nc*T_origin, L_traj, 3] for the model input
#         # This is tricky - we need to expand the temporal dimension
        
#         # For training, we use a simplified approach:
#         # Use the noised target directly as if it were downsampled trajectory input
#         # In practice, you might want to handle this differently
        
#         # Predict the target trajectory
#         pred = self.forward(
#             traj=traj,
#             traj_ids=traj_ids,
#             cond=cond,
#             cond_ids=cond_ids,
#             timesteps=t,
#             guidance=guidance
#         )
        
#         # Compute prediction target (velocity prediction)
#         target = target_traj - noise
        
#         # MSE loss
#         assert pred.shape == target.shape == target_traj.shape
#         terms["mse"] = mean_flat((target - pred) ** 2)
        
#         # Total loss
#         terms["loss"] = terms["mse"].mean()
        
#         # Compute denoised prediction if requested
#         if return_predict:
#             # x_0 = x_t + pred * (1 - t)
#             predict = x_t + pred * (1.0 - t_reshaped)
#             terms["predict"] = predict
#         else:
#             terms["predict"] = None
        
#         return terms
    
#     def sample(
#         self,
#         traj: Tensor,  # [B*Nc*T_origin, L_traj, C_in]
#         traj_ids: Tensor,
#         cond: Tensor,  # [B*Nc*T_latent, L_cond, hidden_size]
#         cond_ids: Tensor,
#         timesteps: list[float],  # Diffusion timesteps for sampling
#         guidance: Tensor | None = None,
#     ) -> Tensor:
#         """
#         Sample trajectories using DDIM or other sampling scheme.
        
#         Args:
#             traj: Initial trajectory (can be noise)
#             traj_ids: Position IDs
#             cond: Conditional features
#             cond_ids: Condition position IDs
#             timesteps: List of timesteps for sampling (descending)
#             guidance: Guidance strength
            
#         Returns:
#             Sampled trajectory [B*Nc*T_latent, L_traj, num_frames*3]
#         """
#         # Start with the downsampled trajectory
#         x_t = self.traj_downsampler(traj)  # [B*Nc*T_latent, L_traj, hidden_size]
        
#         # Initialize output trajectory
#         # For the first step, we need to get the shape right
#         B_latent, L_traj, _ = x_t.shape
        
#         # Sample initial noise in the output space
#         x_t_out = torch.randn(
#             B_latent, L_traj, self.num_output_frames * self.params.traj_out_channels,
#             device=traj.device, dtype=traj.dtype
#         )
        
#         # Iterative denoising
#         for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
#             t_vec = torch.full((B_latent,), t_curr, dtype=traj.dtype, device=traj.device)
            
#             # Predict velocity
#             pred = self.forward(
#                 traj=traj,
#                 traj_ids=traj_ids,
#                 cond=cond,
#                 cond_ids=cond_ids,
#                 timesteps=t_vec,
#                 guidance=guidance
#             )
            
#             # Update: x_{t-1} = x_t + (t_prev - t_curr) * pred
#             x_t_out = x_t_out + (t_prev - t_curr) * pred
        
#         return x_t_out
    
#     @torch.no_grad()
#     def generate(
#         self,
#         cond: Tensor,  # [B*Nc*T_latent, L_cond, hidden_size] - from mvstt
#         cond_ids: Tensor,
#         L_traj: int,  # Number of trajectory tokens
#         num_sampling_steps: int = 50,
#         guidance_scale: float = 1.0,
#         device: torch.device = None,
#     ) -> Tensor:
#         """
#         Generate trajectory from scratch (for inference).
        
#         Args:
#             cond: Conditional features from mvstt
#             cond_ids: Position IDs for condition
#             L_traj: Number of trajectory tokens to generate
#             num_sampling_steps: Number of diffusion sampling steps
#             guidance_scale: Classifier-free guidance scale
#             device: Device to run on
            
#         Returns:
#             Generated trajectory [B*Nc*T_latent, L_traj, num_frames*3]
#         """
#         if device is None:
#             device = cond.device
        
#         B_latent = cond.shape[0]
        
#         # Create trajectory position IDs
#         traj_ids = torch.arange(L_traj, device=device)[None, :, None].expand(B_latent, -1, -1)
        
#         # Start from pure noise
#         x_t = torch.randn(
#             B_latent, L_traj, self.num_output_frames * self.params.traj_out_channels,
#             device=device, dtype=cond.dtype
#         )
        
#         # Create sampling schedule (linear for simplicity)
#         timesteps = torch.linspace(1.0, 0.0, num_sampling_steps + 1, device=device).tolist()
        
#         # Prepare guidance tensor
#         guidance = None
#         if self.params.guidance_embed and guidance_scale != 1.0:
#             guidance = torch.full((B_latent,), guidance_scale, device=device, dtype=cond.dtype)
        
#         # We need a dummy trajectory input for the downsampler
#         # The downsampler expects [B*Nc*T_origin, L_traj, C_in]
#         # We know B_latent = B*Nc*T_latent, so B*Nc = B_latent / T_latent
#         T_latent = self.params.T_origin // self.params.traj_temporal_compression
#         B_Nc = B_latent // T_latent
#         traj_dummy = torch.zeros(
#             B_Nc * self.params.T_origin, L_traj, self.params.traj_in_channels,
#             device=device, dtype=cond.dtype
#         )
        
#         # Sampling loop
#         for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
#             t_vec = torch.full((B_latent,), t_curr, dtype=cond.dtype, device=device)
            
#             # Predict velocity
#             pred = self.forward(
#                 traj=traj_dummy,
#                 traj_ids=traj_ids,
#                 cond=cond,
#                 cond_ids=cond_ids,
#                 timesteps=t_vec,
#                 guidance=guidance
#             )
            
#             # Update
#             x_t = x_t + (t_prev - t_curr) * pred
        
#         return x_t


# # ==========================================
# # Testing and Validation
# # ==========================================
# if __name__ == "__main__":
#     print("=" * 80)
#     print("Testing Physical DiT")
#     print("=" * 80)
    
#     # Configuration
#     B = 2  # Batch size
#     Nc = 6  # Number of cameras
#     T_latent = 4  # Temporal length after VAE compression
#     T_origin = T_latent * 4  # Original temporal length (before compression)
    
#     L_img = 256  # Number of image tokens (e.g., 16x16 patches)
#     L_pose = 4  # Number of pose tokens
#     L_cond = L_img + L_pose  # Total condition length
#     L_traj = 12  # Number of trajectory tokens to predict
    
#     hidden_size = 1024
#     num_output_frames = 12
    
#     # Create model parameters
#     params = PhysicalDiTParams(
#         traj_in_channels=3,  # xyz
#         traj_out_channels=3,  # xyz
#         hidden_size=hidden_size,
#         num_heads=16,
#         depth=1,  # DoubleStream blocks
#         depth_single_blocks=1,  # SingleStream blocks
#         traj_temporal_compression=4,
#         T_origin=T_origin,  # Add this parameter
#         num_output_frames=num_output_frames,
#         guidance_embed=True,
#     )
    
#     # Create model
#     model = PhysicalDiT(params).cuda()
    
#     print(f"\nModel Configuration:")
#     print(f"  Input trajectory channels: {params.traj_in_channels} (xyz)")
#     print(f"  Output trajectory channels: {params.traj_out_channels} (xyz)")
#     print(f"  Output frames: {params.num_output_frames}")
#     print(f"  Temporal compression: {params.traj_temporal_compression}x")
#     print(f"  Hidden size: {params.hidden_size}")
#     print(f"  DoubleStream blocks: {params.depth}")
#     print(f"  SingleStream blocks: {params.depth_single_blocks}")
    
#     # Test 1: Forward pass
#     print("\n" + "=" * 80)
#     print("Test 1: Forward Pass")
#     print("=" * 80)
    
#     B_full_latent = B * Nc * T_latent
#     B_full_origin = B * Nc * T_origin
    
#     # Create dummy inputs
#     traj_input = torch.randn(B_full_origin, L_traj, 3).cuda()  # xyz coordinates
#     traj_ids = torch.arange(L_traj).reshape(1, L_traj, 1).expand(B_full_latent, -1, -1).cuda()
    
#     # Condition from mvstt (already in hidden dimension)
#     cond_input = torch.randn(B_full_latent, L_cond, hidden_size).cuda()
#     cond_ids = torch.arange(L_cond).reshape(1, L_cond, 1).expand(B_full_latent, -1, -1).cuda()
    
#     # Timesteps
#     timesteps = torch.rand(B_full_latent).cuda()
#     guidance = torch.ones(B_full_latent).cuda() * 1.5
    
#     print(f"Input shapes:")
#     print(f"  Trajectory: {traj_input.shape}")
#     print(f"  Trajectory IDs: {traj_ids.shape}")
#     print(f"  Condition: {cond_input.shape}")
#     print(f"  Condition IDs: {cond_ids.shape}")
#     print(f"  Timesteps: {timesteps.shape}")
#     print(f"  Guidance: {guidance.shape}")
    
#     # Forward pass
#     output = model(
#         traj=traj_input,
#         traj_ids=traj_ids,
#         cond=cond_input,
#         cond_ids=cond_ids,
#         timesteps=timesteps,
#         guidance=guidance
#     )
    
#     expected_shape = (B_full_latent, L_traj, num_output_frames * 3)
#     print(f"\nOutput shape: {output.shape}")
#     print(f"Expected shape: {expected_shape}")
#     assert output.shape == expected_shape, f"Shape mismatch! Got {output.shape}, expected {expected_shape}"
#     print("✅ Forward pass successful!")
    
#     # Test 2: Training losses
#     print("\n" + "=" * 80)
#     print("Test 2: Training Losses")
#     print("=" * 80)
    
#     # Ground truth trajectory (12 frames * 3 channels)
#     target_traj = torch.randn(B_full_latent, L_traj, num_output_frames * 3).cuda()
    
#     # Diffusion time (0 to 1)
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
    
#     print(f"Loss components:")
#     print(f"  MSE loss shape: {losses['mse'].shape}")
#     print(f"  Total loss: {losses['loss'].item():.6f}")
#     print(f"  Prediction shape: {losses['predict'].shape}")
#     print("✅ Training losses computed successfully!")
    
#     # Test 3: Generation
#     print("\n" + "=" * 80)
#     print("Test 3: Trajectory Generation")
#     print("=" * 80)
    
#     generated_traj = model.generate(
#         cond=cond_input,
#         cond_ids=cond_ids,
#         L_traj=L_traj,
#         num_sampling_steps=20,
#         guidance_scale=1.5,
#     )
    
#     print(f"Generated trajectory shape: {generated_traj.shape}")
#     print(f"Expected shape: {expected_shape}")
#     assert generated_traj.shape == expected_shape
#     print("✅ Generation successful!")
    
#     # Test 4: Reshape to individual frames
#     print("\n" + "=" * 80)
#     print("Test 4: Reshape to Individual Frames")
#     print("=" * 80)
    
#     # Reshape output to [B, Nc, T_latent, L_traj, num_frames, 3]
#     output_reshaped = output.reshape(B, Nc, T_latent, L_traj, num_output_frames, 3)
    
#     print(f"Reshaped output: {output_reshaped.shape}")
#     print(f"  Batch size: {B}")
#     print(f"  Number of cameras: {Nc}")
#     print(f"  Temporal length: {T_latent}")
#     print(f"  Trajectory tokens: {L_traj}")
#     print(f"  Output frames: {num_output_frames}")
#     print(f"  Coordinates (xyz): 3")
#     print("✅ Reshape successful!")
    
#     print("\n" + "=" * 80)
#     print("🎉 All Tests Passed!")
#     print("=" * 80)
    
#     # Print model statistics
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f"\nModel Statistics:")
#     print(f"  Total parameters: {total_params:,}")
#     print(f"  Trainable parameters: {trainable_params:,}")
#     print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (fp32)")

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
        if traj.ndim != 3:
            raise ValueError(f"Expected traj to be 3D [B, L, C], got shape {traj.shape}")
        if cond.ndim != 3:
            raise ValueError(f"Expected cond to be 3D [B, L, C], got shape {cond.shape}")
        
        # Step 1: Downsample and project trajectory
        traj = self.traj_downsampler(traj)  # [B*Nc*T_origin, L, 3] -> [B*Nc*T_latent, L, hidden]
        
        # CRITICAL: Ensure traj is 3D after downsampling
        if traj.ndim != 3:
            raise ValueError(f"After downsampling, traj should be 3D, got shape {traj.shape}")
        
        traj = self.traj_in(traj)  # [B*Nc*T_latent, L, hidden]
        
        # CRITICAL: Validate traj shape after projection
        if traj.ndim != 3:
            raise ValueError(f"After traj_in, traj should be 3D, got shape {traj.shape}")
        
        # Step 2: Project condition
        cond = self.cond_in(cond)  # [B*Nc*T_latent, L, hidden]
        
        if cond.ndim != 3:
            raise ValueError(f"After cond_in, cond should be 3D, got shape {cond.shape}")
        
        # Step 3: Time and guidance embeddings
        vec = self.time_in(timestep_embedding(timesteps, 256))  # [B*Nc*T_latent, hidden]
        
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Guidance required for guidance-distilled model")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        
        # Step 4: Generate position embeddings
        # Concatenate position IDs: [B, L_cond+L_traj, 1]
        ids = torch.cat([cond_ids, traj_ids], dim=1)
        pe = self.pe_embedder(ids)  # [B, 1, L_cond+L_traj, pe_dim//2, 2, 2]
        
        # Step 5: DoubleStream blocks
        for block in self.double_blocks:
            traj, cond = block(img=traj, cond=cond, vec=vec, pe=pe)
            
            # Validate shapes remain 3D
            if traj.ndim != 3 or cond.ndim != 3:
                raise ValueError(
                    f"After DoubleStreamBlock, expected 3D tensors, "
                    f"got traj.shape={traj.shape}, cond.shape={cond.shape}"
                )
        
        # Step 6: Concatenate streams
        x = torch.cat([cond, traj], dim=1)  # [B, L_cond+L_traj, hidden]
        
        # Step 7: SingleStream blocks
        for block in self.single_blocks:
            x = block(x, vec=vec, pe=pe)
            
            if x.ndim != 3:
                raise ValueError(f"After SingleStreamBlock, expected 3D tensor, got shape {x.shape}")
        
        # Step 8: Extract trajectory tokens and predict
        traj_out = x[:, cond.shape[1]:, ...]  # [B, L_traj, hidden]
        traj_out = self.final_layer(traj_out, vec)  # [B, L_traj, num_frames*3]
        
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
        print(f"\nOutput shape: {output.shape}")
        print(f"Expected shape: {expected_shape}")
        assert output.shape == expected_shape
        print("✅ Forward pass successful!")
        
    except Exception as e:
        print(f"❌ Forward pass failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Test 2: Training losses
    print("\n" + "=" * 80)
    print("Test 2: Training Losses")
    print("=" * 80)
    
    target_traj = torch.randn(B_full_latent, L_traj, num_output_frames * 3).cuda()
    t = torch.rand(B_full_latent).cuda()
    
    losses = model.training_losses(
        traj=traj_input,
        traj_ids=traj_ids,
        cond=cond_input,
        cond_ids=cond_ids,
        t=t,
        target_traj=target_traj,
        guidance=guidance,
        return_predict=True
    )
    
    print(f"Loss: {losses['loss'].item():.6f}")
    print("✅ Training losses computed successfully!")
    
    # Test 3: Generation
    print("\n" + "=" * 80)
    print("Test 3: Generation")
    print("=" * 80)
    
    generated = model.generate(
        cond=cond_input,
        cond_ids=cond_ids,
        L_traj=L_traj,
        num_sampling_steps=20,
        guidance_scale=1.5,
    )
    
    print(f"Generated shape: {generated.shape}")
    print("✅ Generation successful!")
    
    print("\n" + "=" * 80)
    print("🎉 All Tests Passed!")
    print("=" * 80)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")