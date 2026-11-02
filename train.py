"""
LM training script.

Features:
- Uses ModelConfig from config.py as the only configuration source.
- Reproducible Python, NumPy, PyTorch, CUDA, and data sampling.
- Detailed training, evaluation, timing, throughput, and VRAM logging.
- Per-layer/module parameter and memory diagnostics.
- Random train.bin and val.bin text samples after evaluation.
- Main loss and MTP loss reporting.
- AMP with GradScaler for CUDA FP16.
- Best validation checkpoint saving.

Expected model interface:
    logits, total_loss, main_loss, mtp_loss = model(X, Y)

Dataset format:
    Pre-tokenized uint16 .bin files.
"""

import math
import random
import time
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from model import Model
from config import ModelConfig


# ============================================================
# LOGGING
# ============================================================

LOGGER = logging.getLogger("LM.train")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def section(title):
    LOGGER.info("")
    LOGGER.info("=" * 100)
    LOGGER.info(" %s", title)
    LOGGER.info("=" * 100)


def fmt_time(seconds):
    seconds = max(0.0, float(seconds))

    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, seconds = divmod(seconds, 60)

    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"

    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def fmt_bytes(num_bytes):
    num_bytes = float(num_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"

        num_bytes /= 1024

    return f"{num_bytes:.2f} PB"


# ============================================================
# CUDA / VRAM
# ============================================================

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_vram(device):
    """
    Returns the current CUDA memory statistics.

    allocated:
        Memory occupied by tensors.

    reserved:
        Memory held by PyTorch's caching allocator.

    peak:
        Peak allocated memory since the last reset.

    free:
        Current device memory reported by CUDA.
    """

    if device.type != "cuda":
        return None

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)

    free, total = torch.cuda.mem_get_info(device)

    return {
        "allocated": allocated,
        "reserved": reserved,
        "peak": peak,
        "free": free,
        "total": total,
    }


def log_vram(label, device):
    stats = get_vram(device)

    if stats is None:
        LOGGER.info("%s | VRAM: unavailable (CPU)", label)
        return

    LOGGER.info(
        "%s | "
        "Allocated=%s | "
        "Reserved=%s | "
        "Peak allocated=%s | "
        "Free=%s / %s",
        label,
        fmt_bytes(stats["allocated"]),
        fmt_bytes(stats["reserved"]),
        fmt_bytes(stats["peak"]),
        fmt_bytes(stats["free"]),
        fmt_bytes(stats["total"]),
    )


# ============================================================
# CONFIGURATION
# ============================================================

def get_config():
    """
    Construct the real configuration from config.py.

    No fallback configuration is created.
    """

    config = ModelConfig()

    required = [
        "seed",
        "device",
        "dtype",
        "train_path",
        "val_path",
        "block_size",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "min_lr",
        "warmup_steps",
        "max_iters",
        "eval_interval",
        "eval_iters",
        "weight_decay",
        "betas",
        "eps",
        "model_save_folder",
        "train_samples",
        "train_sample_max_tokens",
    ]

    missing = [
        name for name in required
        if not hasattr(config, name)
    ]

    if missing:
        raise AttributeError(
            "ModelConfig is missing required fields: "
            + ", ".join(missing)
        )

    LOGGER.info(
        "Configuration loaded from: %s",
        ModelConfig.__module__,
    )

    LOGGER.info(
        "Configuration class: %s",
        type(config).__name__,
    )

    LOGGER.info(
        "Configuration object ID: %s",
        id(config),
    )

    return config


def get_mtp_setting(config):
    """
    Detect the MTP configuration field.

    Supported field names:
        mtp
        mtp_heads
        n_mtp
        num_mtp_heads

    The first matching field is used.
    """

    names = (
        "mtp",
        "mtp_heads",
        "n_mtp",
        "num_mtp_heads",
    )

    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)

            if value is None:
                continue

            return name, int(value)

    raise AttributeError(
        "No MTP configuration field found in ModelConfig. "
        "Expected one of: "
        + ", ".join(names)
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    LOGGER.info("Global random seed: %d", seed)

    LOGGER.info(
        "Python / NumPy / PyTorch / CUDA RNGs initialized."
    )


# ============================================================
# DATA
# ============================================================

def load_dataset(path, split):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{split} dataset does not exist: {path}"
        )

    data = np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
    )

    if len(data) <= 1:
        raise ValueError(
            f"{split} dataset is too small: {len(data)} tokens"
        )

    LOGGER.info(
        "%s dataset loaded | Path=%s | Tokens=%s | Size=%s",
        split.upper(),
        path.resolve(),
        f"{len(data):,}",
        fmt_bytes(path.stat().st_size),
    )

    return data


def validate_dataset(data, block_size, split):
    required = block_size + 1

    if len(data) < required:
        raise ValueError(
            f"{split} dataset has {len(data):,} tokens, "
            f"but block_size={block_size} requires at least "
            f"{required:,} tokens."
        )


def get_batch(data, batch_size, block_size, device, rng):
    """
    Sample random next-token prediction batches.

    X:
        Tokens [start : start + block_size]

    Y:
        Tokens [start + 1 : start + block_size + 1]
    """

    max_start = len(data) - block_size

    if max_start <= 0:
        raise ValueError(
            "Dataset is too small for the configured block_size."
        )

    indices = rng.integers(
        low=0,
        high=max_start,
        size=batch_size,
    )

    x = np.stack([
        data[index:index + block_size]
        for index in indices
    ]).astype(np.int64)

    y = np.stack([
        data[index + 1:index + block_size + 1]
        for index in indices
    ]).astype(np.int64)

    X = torch.from_numpy(x).to(
        device=device,
        non_blocking=False,
    )

    Y = torch.from_numpy(y).to(
        device=device,
        non_blocking=False,
    )

    return X, Y


# ============================================================
# TOKENIZER / TEXT SAMPLES
# ============================================================

def load_tokenizer():
    """
    Try common tokenizer locations.

    If no tokenizer is available, sampled token IDs
    will be printed instead.
    """

    candidates = [
        Path("tokenizer/tokenizer.json"),
        Path("tokenizer.json"),
    ]

    for path in candidates:
        if not path.exists():
            continue

        try:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(
                str(path)
            )

            LOGGER.info(
                "Tokenizer loaded: %s",
                path.resolve(),
            )

            return tokenizer

        except Exception as exc:
            LOGGER.warning(
                "Tokenizer loading failed for %s: %s",
                path,
                exc,
            )

    LOGGER.warning(
        "No compatible tokenizer found. "
        "Evaluation samples will print token IDs."
    )

    return None


def decode_tokens(tokens, tokenizer):
    tokens = [int(token) for token in tokens]

    if tokenizer is None:
        return " ".join(map(str, tokens))

    try:
        return tokenizer.decode(
            tokens,
            skip_special_tokens=False,
        )

    except Exception:
        return " ".join(map(str, tokens))


def print_random_samples(
    train_data,
    val_data,
    tokenizer,
    sample_count,
    max_tokens,
    seed,
):
    """
    Print randomly selected snippets from train.bin and val.bin.

    The configured total sample count is split equally:
        train_samples=8 -> 4 train + 4 validation snippets.

    Sampling uses a separate RNG so that evaluation samples
    do not alter the training batch RNG sequence.
    """

    if sample_count <= 0:
        LOGGER.info(
            "Evaluation samples disabled: train_samples=%d",
            sample_count,
        )
        return

    if max_tokens <= 0:
        LOGGER.warning(
            "Evaluation samples disabled: "
            "train_sample_max_tokens=%d",
            max_tokens,
        )
        return

    train_count = sample_count // 2
    val_count = sample_count - train_count

    rng = np.random.default_rng(
        seed + 1234567
    )

    section("RANDOM TRAIN / VALIDATION TEXT SAMPLES")

    LOGGER.info(
        "Configured samples=%d | Train=%d | Validation=%d | "
        "Maximum tokens per snippet=%d",
        sample_count,
        train_count,
        val_count,
        max_tokens,
    )

    def print_split(data, split, count):
        LOGGER.info("")
        LOGGER.info("-" * 100)
        LOGGER.info("%s.BIN SAMPLES", split.upper())
        LOGGER.info("-" * 100)

        for sample_index in range(count):
            available = len(data)

            if available <= 0:
                LOGGER.warning(
                    "%s dataset is empty.",
                    split,
                )
                return

            start = int(
                rng.integers(0, available)
            )

            end = min(
                start + max_tokens,
                available,
            )

            tokens = data[start:end]

            text = decode_tokens(
                tokens,
                tokenizer,
            )

            LOGGER.info(
                "%s sample %d/%d | "
                "Start token=%d | "
                "Sample tokens=%d",
                split.upper(),
                sample_index + 1,
                count,
                start,
                len(tokens),
            )

            print(text, flush=True)

            LOGGER.info("-" * 100)

    print_split(
        train_data,
        "train",
        train_count,
    )

    print_split(
        val_data,
        "val",
        val_count,
    )


# ============================================================
# MODEL DIAGNOSTICS
# ============================================================

def parameter_count(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def parameter_bytes(module):
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
    )


def log_model_structure(model, device):
    """
    Print model architecture and per-module parameter memory.

    This is a pre-training diagnostic.

    Module parameter memory is an estimate based on the
    parameters' actual dtype and size.

    CUDA VRAM is measured globally after model placement.
    """

    section("MODEL ARCHITECTURE AND MEMORY DIAGNOSTICS")

    total_params = parameter_count(model)

    LOGGER.info(
        "Total parameters: %s",
        f"{total_params:,}",
    )

    LOGGER.info(
        "Total parameter memory: %s",
        fmt_bytes(parameter_bytes(model)),
    )

    LOGGER.info(
        "Module diagnostics include all named modules, "
        "including attention, MLP, MoE and expert modules."
    )

    LOGGER.info(
        "Per-module parameter memory is not additive "
        "when modules share parameters."
    )

    for name, module in model.named_modules():
        if not name:
            name = "<root>"

        count = parameter_count(module)

        direct_count = sum(
            parameter.numel()
            for parameter in module.parameters(
                recurse=False
            )
        )

        memory = parameter_bytes(module)

        LOGGER.info(
            "MODULE | %-60s | "
            "Type=%-30s | "
            "Parameters=%12s | "
            "Direct parameters=%12s | "
            "Parameter memory=%s",
            name,
            module.__class__.__name__,
            f"{count:,}",
            f"{direct_count:,}",
            fmt_bytes(memory),
        )

    log_vram(
        "Model structure diagnostic complete",
        device,
    )


def log_startup_config(
    config,
    device,
    dtype,
    mtp_name,
    mtp_value,
):
    section("TRAINING CONFIGURATION")

    values = {
        "Config class": type(config).__name__,
        "Config module": ModelConfig.__module__,
        "Seed": config.seed,
        "Device": device,
        "Dtype": dtype,
        "MTP field": mtp_name,
        "MTP value": mtp_value,
        "MTP enabled": mtp_value > 0,
        "Train dataset": config.train_path,
        "Validation dataset": config.val_path,
        "Block size": config.block_size,
        "Batch size": config.batch_size,
        "Gradient accumulation": config.gradient_accumulation_steps,
        "Tokens per optimizer step": (
            config.batch_size
            * config.gradient_accumulation_steps
            * config.block_size
        ),
        "Maximum iterations": config.max_iters,
        "Warmup steps": config.warmup_steps,
        "Evaluation interval": config.eval_interval,
        "Evaluation batches": config.eval_iters,
        "Learning rate": config.learning_rate,
        "Minimum learning rate": config.min_lr,
        "Weight decay": config.weight_decay,
        "Betas": config.betas,
        "Epsilon": config.eps,
        "Compile": getattr(config, "compile", False),
        "Train samples per evaluation": config.train_samples,
        "Sample maximum tokens": config.train_sample_max_tokens,
        "Checkpoint folder": config.model_save_folder,
    }

    for key, value in values.items():
        LOGGER.info("%-35s : %s", key, value)


# ============================================================
# LOSS HANDLING
# ============================================================

def extract_loss_value(value):
    """
    Convert a scalar tensor or numeric value to float.
    """

    if value is None:
        return 0.0

    if torch.is_tensor(value):
        if value.numel() != 1:
            return None

        return float(value.detach().float().item())

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


def extract_mtp_losses(mtp_loss):
    """
    Extract per-horizon MTP losses where available.

    Supported forms:

        scalar tensor:
            One combined MTP loss.

        list / tuple:
            One loss per prediction horizon.

        dictionary:
            Horizon name -> loss.
    """

    if mtp_loss is None:
        return {}

    if isinstance(mtp_loss, dict):
        results = {}

        for name, value in mtp_loss.items():
            loss_value = extract_loss_value(value)

            if loss_value is not None:
                results[str(name)] = loss_value

        return results

    if isinstance(mtp_loss, (list, tuple)):
        results = {}

        for index, value in enumerate(mtp_loss):
            loss_value = extract_loss_value(value)

            if loss_value is not None:
                results[f"horizon_{index + 1}"] = loss_value

        return results

    value = extract_loss_value(mtp_loss)

    if value is None:
        return {}

    return {
        "combined": value,
    }


def format_mtp_losses(mtp_losses):
    if not mtp_losses:
        return "disabled / unavailable"

    return " | ".join(
        f"{name}={value:.5f}"
        for name, value in mtp_losses.items()
    )


# ============================================================
# LEARNING RATE
# ============================================================

def get_learning_rate(config, step):
    warmup_steps = int(config.warmup_steps)

    if warmup_steps > 0 and step <= warmup_steps:
        return (
            config.learning_rate
            * step
            / warmup_steps
        )

    progress = (
        step - warmup_steps
    ) / max(
        1,
        config.max_iters - warmup_steps,
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    return (
        config.min_lr
        + (
            config.learning_rate
            - config.min_lr
        )
        * (
            1 + math.cos(math.pi * progress)
        )
        / 2
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.inference_mode()
def estimate_loss(
    model,
    config,
    train_data,
    val_data,
    device,
    dtype,
    amp,
    eval_rng,
):
    model.eval()

    losses = {}

    split_data = {
        "train": train_data,
        "val": val_data,
    }

    for split, data in split_data.items():
        total_loss = 0.0
        total_main_loss = 0.0

        mtp_totals = {}

        for _ in range(config.eval_iters):
            X, Y = get_batch(
                data,
                config.batch_size,
                config.block_size,
                device,
                eval_rng,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=amp,
            ):
                _, loss, main_loss, mtp_loss = model(
                    X,
                    Y,
                )

            total_loss += extract_loss_value(loss)
            total_main_loss += extract_loss_value(main_loss)

            for name, value in extract_mtp_losses(
                mtp_loss
            ).items():
                mtp_totals[name] = (
                    mtp_totals.get(name, 0.0)
                    + value
                )

        losses[split] = {
            "loss": total_loss / config.eval_iters,
            "main": total_main_loss / config.eval_iters,
            "mtp": {
                name: value / config.eval_iters
                for name, value in mtp_totals.items()
            },
        }

    model.train()

    return losses


# ============================================================
# TRAINING
# ============================================================

def train_model():
    setup_logging()

    section("LM TRAINING INITIALIZATION")

    startup_time = time.perf_counter()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = get_config()

    seed = int(config.seed)

    set_seed(seed)

    mtp_name, mtp_value = get_mtp_setting(
        config
    )

    if mtp_value < 0:
        raise ValueError(
            f"{mtp_name} must be >= 0, got {mtp_value}"
        )

    # --------------------------------------------------------
    # Device / dtype
    # --------------------------------------------------------

    requested_device = str(config.device)

    if (
        requested_device.startswith("cuda")
        and torch.cuda.is_available()
    ):
        device = torch.device(
            requested_device
        )

    else:
        device = torch.device("cpu")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if config.dtype not in dtype_map:
        raise ValueError(
            f"Unsupported dtype: {config.dtype}"
        )

    dtype = dtype_map[
        config.dtype
    ]

    amp = (
        device.type == "cuda"
        and dtype != torch.float32
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            amp
            and dtype == torch.float16
        ),
    )

    config.model_save_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_startup_config(
        config,
        device,
        dtype,
        mtp_name,
        mtp_value,
    )

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(
            device
        )

        section("GPU INFORMATION")

        LOGGER.info(
            "GPU name: %s",
            properties.name,
        )

        LOGGER.info(
            "GPU capability: %s",
            torch.cuda.get_device_capability(device),
        )

        LOGGER.info(
            "Total VRAM: %s",
            fmt_bytes(properties.total_memory),
        )

        LOGGER.info(
            "CUDA version: %s",
            torch.version.cuda,
        )

        LOGGER.info(
            "PyTorch version: %s",
            torch.__version__,
        )

        log_vram(
            "Before dataset loading",
            device,
        )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    section("DATASET LOADING")

    dataset_start = time.perf_counter()

    train_data = load_dataset(
        config.train_path,
        "train",
    )

    val_data = load_dataset(
        config.val_path,
        "val",
    )

    validate_dataset(
        train_data,
        config.block_size,
        "train",
    )

    validate_dataset(
        val_data,
        config.block_size,
        "val",
    )

    LOGGER.info(
        "Dataset loading completed in %s",
        fmt_time(
            time.perf_counter() - dataset_start
        ),
    )

    # Separate RNGs keep training/evaluation sampling
    # reproducible and independent.
    train_rng = np.random.default_rng(
        seed
    )

    eval_rng = np.random.default_rng(
        seed + 1
    )

    # --------------------------------------------------------
    # Model construction
    # --------------------------------------------------------

    section("MODEL CONSTRUCTION")

    model_start = time.perf_counter()

    log_vram(
        "Before model construction",
        device,
    )

    model = Model(config)

    LOGGER.info(
        "Model constructor completed in %s",
        fmt_time(
            time.perf_counter() - model_start
        ),
    )

    model = model.to(
        device
    )

    synchronize(device)

    LOGGER.info(
        "Model moved to %s",
        device,
    )

    log_vram(
        "After model.to(device)",
        device,
    )

    # --------------------------------------------------------
    # Model structure
    # --------------------------------------------------------

    log_model_structure(
        model,
        device,
    )

    # --------------------------------------------------------
    # Compilation
    # --------------------------------------------------------

    compile_enabled = bool(
        getattr(config, "compile", False)
    )

    if compile_enabled:
        section("MODEL COMPILATION")

        compile_start = time.perf_counter()

        LOGGER.info(
            "torch.compile enabled. "
            "Compilation may happen lazily on the first forward pass."
        )

        model = torch.compile(
            model
        )

        LOGGER.info(
            "torch.compile wrapper created in %s",
            fmt_time(
                time.perf_counter() - compile_start
            ),
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    section("OPTIMIZER INITIALIZATION")

    optimizer_start = time.perf_counter()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
        eps=config.eps,
    )

    LOGGER.info(
        "AdamW initialized in %s",
        fmt_time(
            time.perf_counter() - optimizer_start
        ),
    )

    raw_model = (
        model._orig_mod
        if hasattr(model, "_orig_mod")
        else model
    )

    total_params = parameter_count(
        raw_model
    )

    LOGGER.info(
        "Trainable parameters: %s",
        f"{sum(p.numel() for p in raw_model.parameters() if p.requires_grad):,}",
    )

    LOGGER.info(
        "Total parameters: %s",
        f"{total_params:,}",
    )

    log_vram(
        "After optimizer initialization",
        device,
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = load_tokenizer()

    # --------------------------------------------------------
    # Training preparation
    # --------------------------------------------------------

    section("TRAINING START")

    LOGGER.info(
        "Total optimizer steps: %s",
        f"{config.max_iters:,}",
    )

    LOGGER.info(
        "Tokens per optimizer step: %s",
        f"{config.batch_size * config.gradient_accumulation_steps * config.block_size:,}",
    )

    LOGGER.info(
        "MTP configuration: %s=%d",
        mtp_name,
        mtp_value,
    )

    if mtp_value == 0:
        LOGGER.info(
            "MTP is disabled by configuration."
        )

    else:
        LOGGER.info(
            "MTP is enabled. Individual horizon losses "
            "will be reported when exposed by model.forward."
        )

    LOGGER.info(
        "Training begins after %.2f seconds of initialization.",
        time.perf_counter() - startup_time,
    )

    log_vram(
        "Immediately before training",
        device,
    )

    best_val_loss = float("inf")

    optimizer.zero_grad(
        set_to_none=True
    )

    model.train()

    training_start = time.perf_counter()

    previous_log_time = training_start

    cumulative_tokens = 0

    tokens_per_step = (
        config.batch_size
        * config.gradient_accumulation_steps
        * config.block_size
    )

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    progress_bar = tqdm(
        range(
            1,
            config.max_iters + 1,
        ),
        desc="Training",
        dynamic_ncols=True,
    )

    for step in progress_bar:
        step_start = time.perf_counter()

        lr = get_learning_rate(
            config,
            step,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(
                device
            )

        total_loss = 0.0
        main_loss_total = 0.0

        mtp_loss_totals = {}

        # ----------------------------------------------------
        # Gradient accumulation
        # ----------------------------------------------------

        for accumulation_step in range(
            config.gradient_accumulation_steps
        ):
            X, Y = get_batch(
                train_data,
                config.batch_size,
                config.block_size,
                device,
                train_rng,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=amp,
            ):
                _, loss, main_loss, mtp_loss = model(
                    X,
                    Y,
                )

                scaled_loss = (
                    loss
                    / config.gradient_accumulation_steps
                )

            scaler.scale(
                scaled_loss
            ).backward()

            total_loss += extract_loss_value(
                loss
            )

            main_loss_total += extract_loss_value(
                main_loss
            )

            for name, value in extract_mtp_losses(
                mtp_loss
            ).items():
                mtp_loss_totals[name] = (
                    mtp_loss_totals.get(name, 0.0)
                    + value
                )

        # ----------------------------------------------------
        # Optimizer step
        # ----------------------------------------------------

        scaler.step(
            optimizer
        )

        scaler.update()

        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss /= (
            config.gradient_accumulation_steps
        )

        main_loss_total /= (
            config.gradient_accumulation_steps
        )

        mtp_loss_totals = {
            name: value / config.gradient_accumulation_steps
            for name, value in mtp_loss_totals.items()
        }

        synchronize(device)

        step_time = (
            time.perf_counter()
            - step_start
        )

        cumulative_tokens += tokens_per_step

        tokens_per_second = (
            tokens_per_step
            / max(step_time, 1e-12)
        )

        elapsed = (
            time.perf_counter()
            - training_start
        )

        average_tokens_per_second = (
            cumulative_tokens
            / max(elapsed, 1e-12)
        )

        remaining_steps = (
            config.max_iters
            - step
        )

        eta_seconds = (
            remaining_steps
            * elapsed
            / max(step, 1)
        )

        # ----------------------------------------------------
        # Progress bar
        # ----------------------------------------------------

        progress_bar.set_postfix(
            loss=f"{total_loss:.4f}",
            main=f"{main_loss_total:.4f}",
            lr=f"{lr:.2e}",
            tok_s=f"{tokens_per_second:.0f}",
        )

        # ----------------------------------------------------
        # Detailed logging
        # ----------------------------------------------------

        if step % 10 == 0 or step == 1:
            log_now = time.perf_counter()

            interval_time = (
                log_now
                - previous_log_time
            )

            previous_log_time = log_now

            LOGGER.info("")
            LOGGER.info(
                "STEP %d / %d",
                step,
                config.max_iters,
            )

            LOGGER.info(
                "Loss=%.6f | Main loss=%.6f | "
                "MTP losses=[%s]",
                total_loss,
                main_loss_total,
                format_mtp_losses(
                    mtp_loss_totals
                ),
            )

            LOGGER.info(
                "Learning rate=%.8e | "
                "Gradient scale=%.4f",
                lr,
                scaler.get_scale(),
            )

            LOGGER.info(
                "Step time=%s | "
                "Logging interval=%s | "
                "Elapsed=%s | "
                "ETA=%s",
                fmt_time(step_time),
                fmt_time(interval_time),
                fmt_time(elapsed),
                fmt_time(eta_seconds),
            )

            LOGGER.info(
                "Tokens/step=%s | "
                "Step tokens/sec=%.2f | "
                "Average tokens/sec=%.2f | "
                "Total processed tokens=%s",
                f"{tokens_per_step:,}",
                tokens_per_second,
                average_tokens_per_second,
                f"{cumulative_tokens:,}",
            )

            log_vram(
                f"Step {step} VRAM",
                device,
            )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        should_evaluate = (
            step % config.eval_interval == 0
            or step == config.max_iters
        )

        if should_evaluate:
            section(
                f"EVALUATION AT STEP {step}"
            )

            eval_start = time.perf_counter()

            losses = estimate_loss(
                model,
                config,
                train_data,
                val_data,
                device,
                dtype,
                amp,
                eval_rng,
            )

            eval_time = (
                time.perf_counter()
                - eval_start
            )

            LOGGER.info(
                "Evaluation completed in %s",
                fmt_time(eval_time),
            )

            LOGGER.info(
                "TRAIN | Loss=%.6f | Main=%.6f | MTP=[%s]",
                losses["train"]["loss"],
                losses["train"]["main"],
                format_mtp_losses(
                    losses["train"]["mtp"]
                ),
            )

            LOGGER.info(
                "VALIDATION | Loss=%.6f | Main=%.6f | MTP=[%s]",
                losses["val"]["loss"],
                losses["val"]["main"],
                format_mtp_losses(
                    losses["val"]["mtp"]
                ),
            )

            log_vram(
                "After evaluation",
                device,
            )

            # ------------------------------------------------
            # Random text samples
            # ------------------------------------------------

            print_random_samples(
                train_data=train_data,
                val_data=val_data,
                tokenizer=tokenizer,
                sample_count=config.train_samples,
                max_tokens=config.train_sample_max_tokens,
                seed=seed + step,
            )

            # ------------------------------------------------
            # Save best checkpoint
            # ------------------------------------------------

            current_val_loss = losses["val"]["loss"]

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss

                checkpoint_path = (
                    config.model_save_folder
                    / "best_model.pt"
                )

                torch.save(
                    raw_model.state_dict(),
                    checkpoint_path,
                )

                LOGGER.info(
                    "NEW BEST CHECKPOINT SAVED"
                )

                LOGGER.info(
                    "Path: %s",
                    checkpoint_path.resolve(),
                )

                LOGGER.info(
                    "Best validation loss: %.6f",
                    best_val_loss,
                )

            else:
                LOGGER.info(
                    "No improvement. "
                    "Best validation loss remains %.6f",
                    best_val_loss,
                )

    # ========================================================
    # COMPLETION
    # ========================================================

    section("TRAINING COMPLETE")

    total_training_time = (
        time.perf_counter()
        - training_start
    )

    LOGGER.info(
        "Total training time: %s",
        fmt_time(total_training_time),
    )

    LOGGER.info(
        "Total initialization time: %s",
        fmt_time(
            training_start - startup_time
        ),
    )

    LOGGER.info(
        "Total elapsed time: %s",
        fmt_time(
            time.perf_counter() - startup_time
        ),
    )

    LOGGER.info(
        "Total optimizer steps: %s",
        f"{config.max_iters:,}",
    )

    LOGGER.info(
        "Total processed tokens: %s",
        f"{cumulative_tokens:,}",
    )

    LOGGER.info(
        "Average training throughput: %.2f tokens/sec",
        cumulative_tokens
        / max(total_training_time, 1e-12),
    )

    LOGGER.info(
        "Best validation loss: %.6f",
        best_val_loss,
    )

    log_vram(
        "Final VRAM",
        device,
    )

    return model, config


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_model()
