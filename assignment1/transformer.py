import torch
from torch import nn
from .linear import Linear
from .embedding import Embedding
from .rope import RotaryPositionalEmbedding
from .attention import MultiHeadSelfAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rpe = RotaryPositionalEmbedding(
            self.d_model // num_heads, theta, max_seq_len, device=device
        )
        self.multihead_attn = MultiHeadSelfAttention(
            d_model, num_heads, self.rpe, device=device, dtype=dtype
        )
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.multihead_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TranformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.linear(self.norm(x))
        return x
