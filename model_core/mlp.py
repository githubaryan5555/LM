import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=True):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=bias)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=bias)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)
