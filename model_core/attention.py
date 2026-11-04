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
        
        # Compression dimensions
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        

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
        self.o_proj = nn.Linear(
            self.n_head * self.v_head_dim,
            self.n_embd,
            bias=config.bias
        )

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE
        self.rope = RotaryEmbedding(self.qk_rope_head_dim,config.block_size)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            )
        )

        
        def forward(self, x):
            B, T, C = x.size()
  
    # --------------------------------------------------
    # KV compression
    # --------------------------------------------------

            kv_compressed, k_rope = torch.split(
                self.wkv_a(x),
                [self.kv_lora_rank, self.qk_rope_head_dim],
                dim=-1
            )

            kv_compressed = self.kv_norm(kv_compressed)

    # --------------------------------------------------
    # Query compression
    # --------------------------------------------------

            q_compressed = self.q_norm(self.wq_a(x))

    # --------------------------------------------------
    # Query decompression
    # --------------------------------------------------

            q_nope = self.wq_b(q_compressed).view(
                B, T, self.n_head, self.qk_nope_head_dim
            ).transpose(1, 2)

            q_rope = self.wq_pe(q_compressed).view(
                B, T, self.n_head, self.qk_rope_head_dim
            ).transpose(1, 2)

    # --------------------------------------------------
    # KV decompression
    # --------------------------------------------------

            kv = self.wkv_b(kv_compressed).view(
                B, T, self.n_head,
                self.qk_nope_head_dim + self.v_head_dim
            )

            k_nope, v = torch.split(
                kv,
               [self.qk_nope_head_dim, self.v_head_dim],
                dim=-1
            )

            k_nope = k_nope.transpose(1, 2)
            v = v.transpose(1, 2)

    # --------------------------------------------------
    # Positional key
    # --------------------------------------------------

            k_rope = k_rope.unsqueeze(2).expand(
                B, T, self.n_head, self.qk_rope_head_dim
            ).transpose(1, 2)

    # --------------------------------------------------
    # RoPE
    # --------------------------------------------------

            cos, sin = self.rope(x, T)

            q_rope = apply_rope(q_rope, cos, sin)
            k_rope = apply_rope(k_rope, cos, sin)

    # --------------------------------------------------
    # Attention
    # --------------------------------------------------

            q = torch.cat([q_nope, q_rope], dim=-1)
            k = torch.cat([k_nope, k_rope], dim=-1)
  
            scores = torch.matmul(
                q, k.transpose(-2, -1)
            ) * self.softmax_scale

            scores = scores.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0,
                float('-inf')
            )

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            out = torch.matmul(attn_weights, v)

    # --------------------------------------------------
    # Output projection
    # --------------------------------------------------

                out = out.transpose(1, 2).contiguous().view(
                B, T, self.n_head * self.v_head_dim
            )

            out = self.resid_dropout(self.o_proj(out))

         
        return out

    
        
