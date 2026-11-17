
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding, apply_rope


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head

        # GQA configuration
        self.n_kv_head = getattr(config, "n_kv_head", 2)
        self.head_dim = self.n_embd // self.n_head
        self.rope_dim = config.rope_dim

        assert self.n_head % self.n_kv_head == 0
        assert self.rope_dim <= self.head_dim
        assert self.rope_dim % 2 == 0

        # Standard GQA projections
        self.q_proj = nn.Linear(
            self.n_embd,
            self.n_head * self.head_dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            self.n_embd,
            self.n_kv_head * self.head_dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            self.n_embd,
            self.n_kv_head * self.head_dim,
            bias=False
        )

        self.o_proj = nn.Linear(
            self.n_head * self.head_dim,
            self.n_embd,
            bias=config.bias
        )

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE
        self.rope = RotaryEmbedding(self.rope_dim, config.block_size)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ).view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape into heads
        q = q.view(
            B, T, self.n_head, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T, self.n_kv_head, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T, self.n_kv_head, self.head_dim
        ).transpose(1, 2)

        # Apply RoPE to the first rope_dim dimensions
        cos, sin = self.rope(x, T)

        q_rope = apply_rope(q[..., :self.rope_dim], cos, sin)
        k_rope = apply_rope(k[..., :self.rope_dim], cos, sin)

        q = torch.cat([q_rope, q[..., self.rope_dim:]], dim=-1)
        k = torch.cat([k_rope, k[..., self.rope_dim:]], dim=-1)

        # GQA attention
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.causal_mask[:, :, :T, :T],
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
            enable_gqa=True
        )

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(
            B, T, self.n_head * self.head_dim
        )

        # Output projection
        out = self.resid_dropout(self.o_proj(out))

        return out
