from numpy import float32
import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
        device: torch.Tensor | None = None,
    ):
        super().__init__()
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k 必须是正偶数。")
        if theta <= 0:
            raise ValueError("theta 必须大于 0。")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len 必须大于 0。")
        self.d_k = d_k

        # 每对相邻分量的旋转频率为d_k/2
        pair_dims = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inv_freq = theta ** (-pair_dims / d_k)

        # 所有预计算位置
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError("输入最后一个维度必须等于 d_k。")
        in_dtype = x.dtype

        # 使用实际位置查表，支持非零起点的位置
        token_positions = token_positions.to(device=self.cos_cache.device, dtype=torch.long)
        cos, sin = self.cos_cache[token_positions],self.sin_cache[token_positions]
        # 当多个注意力头共享位置时，补上广播维度，例如 [B, T, d_k/2] → [B, 1, T, d_k/2]
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        # 在float32中进行旋转计算
        x = x.to(torch.float32)
        # 相邻分量配对
        x_even = x[...,0::2]
        x_odd = x[...,1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        
        # 恢复为原来的交错顺序
        result = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(start_dim=-2)
        return result.to(in_dtype)