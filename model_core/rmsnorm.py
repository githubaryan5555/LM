import torch
import torch.nn as nn


"""
Root Mean Square Normalization (RMSNorm)

Normalizes input vectors using their root mean square
and applies a learnable scaling parameter.

RMSNorm:
    Input:
        x: [batch_size, sequence_length, ndim]

    Output:
        normalized_x: [batch_size, sequence_length, ndim]
"""


class RMSNorm(nn.Module):
    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)
