from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:

    # ==================== MODEL VARIABLES ====================

    # Core architecture
    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 8

    # Multi-head latent attention
    kv_lora_rank: int = 64
    q_lora_rank: int = 128
    rope_dim: int = 32

    # Mixture of experts
    n_experts: int = 2
    n_experts_per_token: int = 2
    expert_intermediate_size: int = 256
    shared_expert_intermediate_size: int = 256
    use_shared_expert: bool = True

    # Multi-token prediction
    mtp_num_heads: int = 2

    # ==================== TRAIN VARIABLES ====================

    # Training schedule
    compile: bool = True
    learning_rate: float = 3e-4
    max_iters: int = 1024
    warmup_steps: int = 128
    min_lr: float = 3e-5
    eval_interval: int = 128
    eval_iters: int = 256
    batch_size: int = 64
    gradient_accumulation_steps: int = 32
    train_samples: int = 8
    train_sample_max_tokens: int = 512
    # Optimizer
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1
    eps: float = 1e-8

    # Re regularization and auxiliary losses
    dropout: float = 0.0
    bias: bool = False
    aux_loss_weight: float = 0.0005
    mtp_loss_weight: float = 0.25
    device: str = "cuda"
    dtype: str = "float16"
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
