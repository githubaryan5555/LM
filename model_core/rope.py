import torch
import torch.nn as nn


"""
Rotary Positional Embeddings (RoPE)

Generates rotary frequency embeddings and applies them
to input vectors to encode positional information.

RotaryEmbedding:
    Input:
        x: [batch_size, num_heads, sequence_length, head_dim]
        seq_len: Optional sequence length

    Output:
        cos: [sequence_length, head_dim // 2]
        sin: [sequence_length, head_dim // 2]

apply_rope:
    Input:
        x: [batch_size, num_heads, sequence_length, head_dim]
        cos: [sequence_length, head_dim // 2]
        sin: [sequence_length, head_dim // 2]

    Output:
        rotated_x: [batch_size, num_heads, sequence_length, head_dim]
"""


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]

        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos, sin

def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
