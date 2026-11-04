
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model_core.rmsnorm import RMSNorm
from model_core.attention import MultiHeadLatentAttention
from model_core.moe import MoELayer
from model_core.mtp import MultiTokenPredictionHead


class ExpertPredictor(nn.Module):
    """
    Purpose:
        Predict the next MoE layer's selected experts.

    Inputs:
        Previous layer hidden states.
        Previous layer's full router probabilities.
        Previous layer's selected experts.
        Layer transition information.

    Outputs:
        One prediction logit per expert.
    """

    def __init__(self, config):
        super().__init__()

        self.n_experts = config.n_experts
        self.n_layers = config.n_layer
        self.n_embd = config.n_embd

        input_size = (
            self.n_embd
            + self.n_experts
            + self.n_experts
            + self.n_layers - 1
        )

        hidden_size = max(64, min(256, self.n_embd // 2))

        self.hidden_norm = nn.LayerNorm(self.n_embd)

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.n_experts),
        )

    def forward(
        self,
        hidden_states,
        router_probabilities,
        selected_experts,
        transition_features,
    ):

        hidden_states = self.hidden_norm(hidden_states)

        features = torch.cat(
            [
                hidden_states,
                router_probabilities,
                selected_experts,
                transition_features,
            ],
            dim=-1,
        )

        return self.network(features)


class ModelBlock(nn.Module):
    """
    Purpose:
        Execute MLA and MoE and return actual routing information.

    Outputs:
        Updated hidden states.
        Selected expert indices.
        Full router probabilities.
    """

    def __init__(self, config):
        super().__init__()

        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = MultiHeadLatentAttention(config)

        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MoELayer(config)

    def forward(self, x):

        x = x + self.attn(self.ln_1(x))

        normalized = self.ln_2(x)

        batch_size, seq_len, hidden_size = normalized.shape

        flat = normalized.reshape(-1, hidden_size)

        # Calculate router logits exactly once.
        router_logits = (
            self.mlp.router.proj(flat)
            + self.mlp.router.expert_bias
        )

        # Full probability distribution over all experts.
        router_probabilities = F.softmax(
            router_logits,
            dim=-1,
        )

        # Actual top-k routing.
        top_k_logits, router_indices = torch.topk(
            router_logits,
            self.mlp.top_k,
            dim=-1,
        )

        routing_weights = F.softmax(
            top_k_logits,
            dim=-1,
        )

        # Pass the exact same routing decision to MoE.
        x = x + self.mlp(
            normalized,
            routing=(routing_weights, router_indices),
        )

        return (
            x,
            router_indices,
            router_probabilities.detach(),
        )


class Model(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Token and position embeddings
        self.wte = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        self.wpe = nn.Embedding(
            config.block_size,
            config.n_embd,
        )

        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.h = nn.ModuleList([
            ModelBlock(config)
            for _ in range(config.n_layer)
        ])

        # Final norm
        self.ln_f = RMSNorm(config.n_embd)

        # Language modeling head
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # MTP heads
        if config.mtp_num_heads > 0:
            self.mtp_heads = nn.ModuleList([
                MultiTokenPredictionHead(config, depth)
                for depth in range(1, config.mtp_num_heads + 1)
            ])
        else:
            self.mtp_heads = None

        # Expert predictor
        if config.n_layer > 1:
            self.expert_predictor = ExpertPredictor(config)
        else:
            self.expert_predictor = None

        # Predictor debugging
        self.predictor_loss_weight = 0.1
        self.predictor_print_interval = 100
        self.predictor_steps = 0

        # Initialize weights
        self.apply(self._init_weights)

        # Special initialization for residual projections
        for pn, p in self.named_parameters():
            if (
                pn.endswith("o_proj.weight")
                or pn.endswith("down_proj.weight")
            ):
                nn.init.normal_(
                    p,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(self, idx, targets=None):

        device = idx.device

        batch_size, seq_len = idx.size()

        assert seq_len <= self.config.block_size

        # Token and position embeddings
        pos = torch.arange(
            0,
            seq_len,
            dtype=torch.long,
            device=device,
        )

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)

        x = self.drop(tok_emb + pos_emb)

        # Predictor statistics
        predictor_losses = []
        predictor_overlaps = []
        predictor_exact_matches = []

        previous_hidden = None
        previous_probabilities = None
        previous_selected = None

        # Transformer blocks
        for layer_index, block in enumerate(self.h):

            x, current_indices, current_probabilities = block(x)

            # Train predictor using previous layer information.
            if (
                self.training
                and targets is not None
                and self.expert_predictor is not None
                and previous_hidden is not None
            ):

                number_of_tokens = current_indices.size(0)

                # Previous layer selected experts.
                previous_selected_features = F.one_hot(
                    previous_selected,
                    num_classes=self.config.n_experts,
                ).sum(dim=-1).clamp(max=1).float()

                # Actual current layer selected experts.
                actual_targets = F.one_hot(
                    current_indices,
                    num_classes=self.config.n_experts,
                ).sum(dim=-1).clamp(max=1).float()

                # Transition identity.
                transition_features = F.one_hot(
                    torch.full(
                        (number_of_tokens,),
                        layer_index - 1,
                        dtype=torch.long,
                        device=device,
                    ),
                    num_classes=self.config.n_layer - 1,
                ).float()

                # Predictor input.
                predictor_input_hidden = (
                    previous_hidden.detach()
                    .reshape(-1, self.config.n_embd)
                    .float()
                )

                predictor_input_probabilities = (
                    previous_probabilities.detach().float()
                )

                predictor_input_selected = (
                    previous_selected_features.detach()
                )

                # Predict next layer.
                prediction_logits = self.expert_predictor(
                    predictor_input_hidden,
                    predictor_input_probabilities,
                    predictor_input_selected,
                    transition_features,
                )

                # Account for the smaller number of positive labels.
                positive_weight = torch.tensor(
                    (
                        self.config.n_experts
                        - self.config.n_experts_per_token
                    ) / self.config.n_experts_per_token,
                    device=device,
                    dtype=prediction_logits.dtype,
                )

                predictor_loss = F.binary_cross_entropy_with_logits(
                    prediction_logits,
                    actual_targets,
                    pos_weight=positive_weight,
                )

                predictor_losses.append(predictor_loss)

                # Predicted top-k experts.
                predicted_indices = torch.topk(
                    prediction_logits,
                    self.config.n_experts_per_token,
                    dim=-1,
                ).indices

                predicted_targets = F.one_hot(
                    predicted_indices,
                    num_classes=self.config.n_experts,
                ).sum(dim=-1).float()

                # Top-k overlap.
                overlap = (
                    (predicted_targets * actual_targets)
                    .sum(dim=-1)
                    .div(self.config.n_experts_per_token)
                    .mean()
                )

                # Exact match.
                exact_match = (
                    (predicted_targets == actual_targets)
                    .all(dim=-1)
                    .float()
                    .mean()
                )

                predictor_overlaps.append(overlap)
                predictor_exact_matches.append(exact_match)

            # Save current layer information for next transition.
            previous_hidden = x.detach()

            previous_probabilities = (
                current_probabilities.detach()
            )

            previous_selected = current_indices.detach()

        # Final norm
        x = self.ln_f(x)

        # Main language modeling head
        main_logits = self.lm_head(x)

        main_loss = None

        if targets is not None:

            main_loss = F.cross_entropy(
                main_logits.reshape(-1, main_logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        # MTP
        mtp_loss = None

        if self.mtp_heads is not None and targets is not None:

            mtp_losses = []
            current_hidden = x

            for depth, mtp_head in enumerate(
                self.mtp_heads,
                1,
            ):

                if seq_len > depth:

                    future_indices = idx[:, depth:]

                    future_embeds = self.wte(future_indices)

                    if future_embeds.size(1) < current_hidden.size(1):

                        pad_size = (
                            current_hidden.size(1)
                            - future_embeds.size(1)
                        )

                        padding = torch.zeros(
                            batch_size,
                            pad_size,
                            self.config.n_embd,
                            device=device,
                            dtype=future_embeds.dtype,
                        )

                        future_embeds = torch.cat(
                            [future_embeds, padding],
                            dim=1,
                        )

                    elif future_embeds.size(1) > current_hidden.size(1):

                        future_embeds = future_embeds[
                            :, :current_hidden.size(1)
                        ]

                    current_hidden = mtp_head(
                        current_hidden,
                        future_embeds,
                    )

                    mtp_logits = self.lm_head(current_hidden)

                    if seq_len > depth + 1:

                        shift_logits = mtp_logits[
                            ..., :-(depth + 1), :
                        ].contiguous()

                        shift_labels = targets[
                            ..., depth + 1:
                        ].contiguous()

                        if shift_labels.numel() > 0:

                            mtp_loss_single = F.cross_entropy(
                                shift_logits.reshape(
                                    -1,
                                    shift_logits.size(-1),
                                ),
                                shift_labels.reshape(-1),
                                ignore_index=-1,
                            )

                            mtp_losses.append(mtp_loss_single)

            if mtp_losses:
                mtp_loss = torch.stack(mtp_losses).mean()

        # Predictor loss and debugging
        predictor_loss = None

        if predictor_losses:

            predictor_loss = torch.stack(
                predictor_losses
            ).mean()

            predictor_overlap = torch.stack(
                predictor_overlaps
            ).mean()

            predictor_exact = torch.stack(
                predictor_exact_matches
            ).mean()

            self.predictor_steps += 1

            if (
                self.predictor_steps
                % self.predictor_print_interval == 0
            ):

                print("\n" + "=" * 65)
                print("EXPERT PREDICTOR DEBUG")
                print("=" * 65)

                print(
                    f"Training steps:       {self.predictor_steps}"
                )

                print(
                    f"Predictor loss:       {predictor_loss.detach().item():.6f}"
                )

                print(
                    f"Top-k overlap:        {predictor_overlap.detach().item() * 100:.2f}%"
                )

                print(
                    f"Exact match:          {predictor_exact.detach().item() * 100:.2f}%"
                )

                print(
                    f"Layer transitions:    {len(predictor_losses)}"
                )

                print("=" * 65 + "\n")

        # Combine losses
        if targets is not None:

            total_loss = main_loss

            if mtp_loss is not None:
                total_loss = (
                    total_loss
                    + self.config.mtp_loss_weight * mtp_loss
                )

            if predictor_loss is not None:
                total_loss = (
                    total_loss
                    + self.predictor_loss_weight * predictor_loss
                )

            return (
                main_logits,
                total_loss,
                main_loss,
                mtp_loss,
            )

        return main_logits[:, [-1], :], None, None, None

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
    ):

        for _ in range(max_new_tokens):

            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )

            logits, _, _, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:

                v, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1)),
                )

                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(
                probs,
                num_samples=1,
            )

            idx = torch.cat(
                (idx, idx_next),
                dim=1,
            )

        return idx
