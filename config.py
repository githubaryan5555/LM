from dataclasses import dataclass
from pathlib import Path

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
    train_samples: int = None
    train_sample_max_tokens: int = None
    # Optimizer
    betas: tuple = None
    weight_decay: float = None
    eps: float = None

    # Re regularization and auxiliary losses
    dropout: float = None
    bias: bool = None
    aux_loss_weight: float = None
    mtp_loss_weight: float = None

    # ==================== GENERATION VARIABLES ====================

    # Text generation settings
    max_generated_tokens: int = 1024
    samples: int = 16
    temperature: float = 0.8
    top_k: int = 64

   
   # ============OTHERS========

   # Model name
   model_family_name: str = "5555LM"
   model_version: str = "Test1"

   # Paths
   model_save_folder: Path = Path("ckpt_out")
   tokenizer_path: Path = Path("tokenizer/tokenizer.json")
   train_path: Path = Path("datasets/train.bin") 
   val_path: Path = Path("datasets/val.bin")
