import torch
import torch.nn as nn


"""
Purpose:
    Applies rotary positional encoding to input vectors.

Input:
    x: [batch_size, num_heads, sequence_length, head_dim]
    cos: [sequence_length, head_dim // 2]
    sin: [sequence_length, head_dim // 2]

Output:
    rotated_x: [batch_size, num_heads, sequence_length, head_dim]
"""


class RMSNorm(nn.Module):
    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)
