import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import SwiGLU


class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.n_experts_per_token

        self.proj = nn.Linear(config.n_embd, self.n_experts, bias=False)

        # Auxiliary-loss-free load balancing.
        self.register_buffer("expert_bias", torch.zeros(self.n_experts))
        self.register_buffer(
            "expert_usage_ema",
            torch.full((self.n_experts,), 1.0 / self.n_experts),
        )

        self.usage_ema_decay = 0.9
        self.bias_update_rate = 0.01
        self.zero_assignment_boost = 1.5
        self.max_expert_bias = 1.0

    def forward(self, x):
        logits = self.proj(x) + self.expert_bias
        top_k_logits, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.sigmoid(top_k_logits)
        return weights, indices

    @torch.no_grad()
    def update_load_balance(self, assignment_counts, total_assignments):
        rates = assignment_counts.to(self.expert_usage_ema.dtype) / total_assignments
        self.expert_usage_ema.mul_(self.usage_ema_decay).add_(
            rates,
            alpha=1.0 - self.usage_ema_decay,
        )

        correction = (1.0 / self.n_experts) - self.expert_usage_ema
        boost_zero_underused = (assignment_counts == 0) & (correction > 0)
        correction = torch.where(
            boost_zero_underused,
            correction * self.zero_assignment_boost,
            correction,
        )

        self.expert_bias.add_(correction, alpha=self.bias_update_rate)
        self.expert_bias.sub_(self.expert_bias.mean())
        self.expert_bias.clamp_(-self.max_expert_bias, self.max_expert_bias)


class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.top_k = config.n_experts_per_token
        self.n_embd = config.n_embd

        self.router = Router(config)

        self.experts = nn.ModuleList([
            SwiGLU(
                config.n_embd,
                config.expert_intermediate_size,
                config.n_embd,
                config.bias,
            )
            for _ in range(config.n_experts)
        ])

        if config.use_shared_expert:
            self.shared_expert = SwiGLU(
                config.n_embd,
                config.shared_expert_intermediate_size,
                config.n_embd,
                config.bias,
            )
        else:
            self.shared_expert = None

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.reshape(-1, hidden_size)

        routing_weights, top_k_indices = self.router(x_flat)

        # Counts top-k assignments, not unique tokens.
        assignment_counts = torch.bincount(
            top_k_indices.reshape(-1),
            minlength=self.n_experts,
        )

        if self.training and top_k_indices.numel() > 0:
            self.router.update_load_balance(
                assignment_counts,
                top_k_indices.numel(),
            )

        output = torch.zeros_like(x_flat)

        for expert_idx, expert in enumerate(self.experts):
            assignment_mask = top_k_indices == expert_idx
            expert_mask = assignment_mask.any(dim=-1)

            # Each selected expert contributes its routing weight for that token.
            expert_weights = (
                routing_weights
                * assignment_mask.to(routing_weights.dtype)
            ).sum(dim=-1)

            expert_input = x_flat[expert_mask]
            expert_output = expert(expert_input)

            output[expert_mask] += (
                expert_output * expert_weights[expert_mask].unsqueeze(-1)
            )

        if self.shared_expert is not None:
            output = output + self.shared_expert(x_flat)

        return output.reshape(batch_size, seq_len, hidden_size)
