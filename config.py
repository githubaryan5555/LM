from dataclasses import dataclass

@dataclass
class ModelConfig:

    # ==================== MODEL VARIABLES ====================

    # Core architecture
    vocab_size: int = None
    block_size: int = None
    n_layer: int = None
    n_embd: int = None
    n_head: int = None

    # Multi-head latent attention
    kv_lora_rank: int = None
    q_lora_rank: int = None
    rope_dim: int = None

    # Mixture of experts
    n_experts: int = None
    n_experts_per_token: int = None
    expert_intermediate_size: int = None
    shared_expert_intermediate_size: int = None
    use_shared_expert: bool = None

    # Multi-token prediction
    mtp_num_heads: int = None

    # ==================== TRAIN VARIABLES ====================

    # Training schedule
    learning_rate: float = None
    max_iters: int = None
    warmup_steps: int = None
    min_lr: float = None
    eval_iters: int = None
    batch_size: int = None
    gradient_accumulation_steps: int = None

    # Optimizer
    betas: tuple = None
    weight_decay: float = None
    eps: float = None

    # Regularization and auxiliary losses
    dropout: float = None
    bias: bool = None
    aux_loss_weight: float = None
    mtp_loss_weight: float = None

    # ==================== GENERATION VARIABLES ====================

    # Text generation settings
    model_path: str = None
    max_tokens: int = None
    temperature: float = None
    top_k: int = None
 

   #!!
