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
        self.head_dim = config.n_embd // config.n_head

        # Compression dimensions
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.rope_dim = config.rope_dim

        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.qk_head_dim = (
            self.qk_nope_head_dim + self.qk_rope_head_dim
        )

        
        # KV compression
        self.wkv_a = nn.Linear(
            self.n_embd,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False
        )

        self.kv_norm = RMSNorm(self.kv_lora_rank)

        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_head * (
                self.qk_nope_head_dim + self.v_head_dim
            ),
            bias=False
        )
   
        
        # Query compression
        self.wq_a = nn.Linear(
            self.n_embd,
            self.q_lora_rank,
            bias=False
        )

        self.q_norm = RMSNorm(self.q_lora_rank)

        self.wq_b = nn.Linear(
            self.q_lora_rank,
            self.n_head * self.qk_nope_head_dim,
            bias=False
        )

        
        # RoPE projections
        self.wq_pe = nn.Linear(
            self.q_lora_rank,
            self.n_head * self.qk_rope_head_dim,
            bias=False
        )
        
        # Output projection
        self.o_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=config.bias)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE
        self.rope = RotaryEmbedding(self.rope_dim, config.block_size)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            )
        )

    def forward(self, x):
        B, T, C = x.size()

        # Compression phase
        kv_compressed = self.kv_norm(self.kv_proj(x))
        q_compressed = self.q_proj(x)

        # Decompression phase
        k_content = self.k_decompress(kv_compressed)
        v = self.v_decompress(kv_compressed)
        q_content = self.q_decompress(q_compressed)

        # RoPE components
        k_rope = self.k_rope_proj(x)
        q_rope = self.q_rope_proj(q_compressed)

        # Reshape for multi-head attention
        k_content = k_content.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q_content = q_content.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k_rope = k_rope.view(B, T, self.n_head, self.rope_dim).transpose(1, 2)
        q_rope = q_rope.view(B, T, self.n_head, self.rope_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(x, T)
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)

        # Concatenate content and rope parts
        q = torch.cat([q_content, q_rope], dim=-1)
        k = torch.cat([k_content, k_rope], dim=-1)

        # Attention computation
        self.softmax_scale = self.qk_head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * self.softmax_scale
        
        # Apply causal mask
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        out = self.resid_dropout(self.o_proj(out))

        return out
