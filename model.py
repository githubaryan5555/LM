import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model_core.rmsnorm import RMSNorm
from model_core.attention import MultiHeadLatentAttention
from model_core.moe import MoELayer
from model_core.mtp import MultiTokenPredictionHead


class ModelBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = MultiHeadLatentAttention(config)

        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MoELayer(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x