import torch
from torch import nn
from einops import einsum



def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.weight1 = nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )
        self.weight2 = nn.Parameter(
            torch.empty(d_model, d_ff, device=device, dtype=dtype)
        )
        self.weight3 = nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )
        std = (2 / (d_model + d_ff)) ** 0.5

        for weight in (self.weight1, self.weight2, self.weight3):
            nn.init.trunc_normal_(
                weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(einsum(x, self.weight1, "... D, F D -> ... F"))
        value = einsum(x, self.weight3, "... D, F D -> ... F")
        return einsum(gate * value, self.weight2, "... F , D F -> ... D")
