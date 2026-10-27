import torch
from torch import nn
from einops import einsum, rearrange
from .linear import Linear
from .rope import RotaryPositionalEmbedding


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_c = x.max(dim=dim, keepdim=True).values
    shifted = x - max_c
    exp_value = shifted.exp()
    return exp_value / exp_value.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = einsum(Q, K, "... Lq Dk, ... Lk Dk -> ... Lq Lk") / Q.shape[-1] ** 0.5
    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), float("-inf"))
    attention = softmax(scores, dim=-1)
    return einsum(attention, V, "... Lq Lk, ... Lk Dv -> ... Lq Dv")


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if num_heads <= 0 or d_model <= 0:
            raise ValueError("d_model 和 num_heads 必须为正数。")
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除。")
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        self.d_k = d_model // num_heads
        # 每次投影一次性计算所有头，不逐头循环
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        Q = rearrange(self.q_proj(x), "... L (H Dk) -> ... H L Dk", H=self.num_heads)
        K = rearrange(self.k_proj(x), "... L (H Dk) -> ... H L Dk", H=self.num_heads)
        V = rearrange(self.v_proj(x), "... L (H Dk) -> ... H L Dk", H=self.num_heads)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
            # 对Q K运用旋转编码
            Q, K = self.rope(Q, token_positions), self.rope(K, token_positions)
        causal_mask = torch.ones(
            seq_len, seq_len, device=x.device, dtype=x.dtype
        ).tril()

        # 每个头单独计算注意力
        head_attention = scaled_dot_product_attention(Q, K, V, causal_mask)
        combined_attention = rearrange(head_attention, "... H L Dk -> ... L (H Dk)")
        return self.output_proj(combined_attention)
