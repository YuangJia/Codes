
import os
import functools
import math
from typing import Optional
import logging

DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")
USE_XFORMERS = eval(os.environ.get("USE_XFORMERS", "False"))

import numpy as np
import torch
if not torch.cuda.is_available() or DEVICE_TYPE == "npu":
    import torch_npu
    USE_NPU = True
else:
    # from flash_attn import flash_attn_func
    USE_NPU = False
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
if USE_XFORMERS:
    try:
        import xformers.ops
    except BaseException as e:
        logging.warning(f"Import xformers got {e}, will disable xformers")
        USE_XFORMERS = False
else:
    logging.info(f"USE_XFORMERS={USE_XFORMERS}")
from einops import rearrange
verbose = False

def approx_gelu(): return nn.GELU(approximate="tanh")

class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (1,2,2).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(1, 2, 2),
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        if np.prod(x.shape[-3:]) > np.prod([33, 112, 200]) and USE_NPU:
            # NOTE: conv3d on NPU cannot take too large batch sizes.
            x = torch.cat([self.proj(_x) for _x in x.chunk(2, dim=0)], dim=0)
        else:
            x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            # x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
            x = x.flatten(3).permute(0, 2, 3, 1)  # BCTHW -> B, T, (H*W), C
        return x