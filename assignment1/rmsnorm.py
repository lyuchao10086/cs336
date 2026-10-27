import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean = x.square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(mean + self.eps)
        # 缩放参数自动广播到所有 batch 和序列位置
        result = normalized * self.weight
        return result.to(in_dtype)
