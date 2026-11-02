# train.py

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from model import Model
from config import ModelConfig


# ============================================================
# Logging utilities
# ============================================================

GIB = 1024 ** 3


def format_time(seconds):
    seconds = max(0, int(seconds))

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"

    if minutes:
        return f"{minutes}m {seconds:02d}s"

    return f"{seconds}s"


def get_vram():
    if not torch.cuda.is_available():
        return None

    free, total = torch.cuda.mem_get_info()

    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "free": free,
        "total": total,
    }


def print_vram(label):
    memory = get_vram()

    if memory is None:
        print(f"[VRAM] {label}: CUDA unavailable")
        return

    print(
        f"[VRAM] {label}\n"
        f"  Allocated:      {memory['allocated'] / GIB:.3f} GiB\n"
        f"  Reserved:       {memory['reserved'] / GIB:.3f} GiB\n"
        f"  Peak allocated: {memory['peak_allocated'] / GIB:.3f} GiB\n"
        f"  Peak reserved:  {memory['peak_reserved'] / GIB:.3f} GiB\n"
        f"  Free:           {memory['free'] / GIB:.3f} GiB\n"
        f"  Total:          {memory['total'] / GIB:.3f} GiB"
    )


def seed_everything(seed):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"[SEED] Using config.seed = {seed}")


def get_lr(config, step):
    if config.warmup_steps > 0 and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps

    progress = (
        step - config.warmup_steps
    ) / max(1, config.max_iters - config.warmup_steps)

    progress = min(1.0, max(0.0, progress))

    return config.min_lr + (
        config.learning_rate - config.min_lr
    ) * (1 + math.cos(math.pi * progress)) / 2


def get_model_loss(output):
    """
    Expected model output:
        logits, total_loss, main_loss, mtp_loss

    MTP loss may be:
        None
        scalar tensor
        list/tuple of scalar tensors
        dictionary of scalar tensors
    """

    if not isinstance(output, (tuple, list)):
        raise TypeError(
            "Model output must be a tuple/list. "
            "Expected (logits, total_loss, main_loss, mtp_loss)."
        )

    if len(output) < 2:
        raise ValueError("Model output does not contain a loss.")

    logits = output[0]
    total_loss = output[1]

    main_loss = output[2] if len(output) > 2 else total_loss
    mtp_loss = output[3] if len(output) > 3 else None

    return logits, total_loss, main_loss, mtp_loss


def normalize_mtp_losses(mtp_loss):
    if mtp_loss is None:
        return {}

    if isinstance(mtp_loss, dict):
        return mtp_loss

    if isinstance(mtp_loss, (tuple, list)):
        return {
            f"mtp_{index + 1}": loss
            for index, loss in enumerate(mtp_loss)
        }

    return {"mtp": mtp_loss}


def tensor_to_float(value):
    if value is None:
        return None

    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())

    return float(value)


def print_loss_summary(losses):
    print(
        f"  Main loss:  {losses['main']:.6f}\n"
        f"  Total loss: {losses['total']:.6f}"
    )

    if losses["mtp"]:
        for name, value in losses["mtp"].items():
            print(f"  {name}: {value:.6f}")
    else:
        print("  MTP: DISABLED")


# ============================================================
# Token dataset
# ============================================================

class TokenDataset:

    def __init__(self, path, block_size, device, name):
        self.path = Path(path)
        self.block_size = block_size
        self.device = device
        self.name = name

        if not self.path.exists():
            raise FileNotFoundError(
                f"{name} dataset not found: {self.path}"
            )

        self.data = np.memmap(
            self.path,
            dtype=np.uint16,
            mode="r",
        )

        minimum_length = block_size + 2

        if len(self.data) < minimum_length:
            raise ValueError(
                f"{name} dataset contains {len(self.data)} tokens. "
                f"At least {minimum_length} are required."
            )

        print(
            f"[DATA] {name}\n"
            f"  Path: {self.path.resolve()}\n"
            f"  Tokens: {len(self.data):,}\n"
            f"  Size: {self.data.nbytes / 1024**2:.2f} MiB"
        )

    def get_batch(self, batch_size):
        max_start = len(self.data) - self.block_size - 1

        indices = np.random.randint(
            0,
            max_start + 1,
            size=batch_size,
        )

        x = np.stack([
            self.data[i:i + self.block_size]
            for i in indices
        ]).astype(np.int64)

        y = np.stack([
            self.data[i + 1:i + self.block_size + 1]
            for i in indices
        ]).astype(np.int64)

        return (
            torch.from_numpy(x).to(self.device),
            torch.from_numpy(y).to(self.device),
        )

    def get_random_snippets(self, count, max_tokens):
        """
        Random contiguous token snippets.
        They may cross sentence boundaries because bin files
        do not preserve explicit sentence-boundary metadata.
        """

        length = min(max_tokens, len(self.data) - 1)

        max_start = len(self.data) - length

        starts = np.random.randint(
            0,
            max_start + 1,
            size=count,
        )

        snippets = []

        for start in starts:
            snippet = np.asarray(
                self.data[start:start + length],
                dtype=np.int64,
            ).copy()

            snippets.append(snippet)

        return snippets


# ============================================================
# Tokenizer
# ============================================================

def load_tokenizer(config):
    tokenizer_path = getattr(config, "tokenizer_path", None)

    if tokenizer_path is None:
        print("[TOKENIZER] No tokenizer_path in config.")
        return None

    tokenizer_path = Path(tokenizer_path)

    if not tokenizer_path.exists():
        print(
            f"[TOKENIZER] Not found: {tokenizer_path}\n"
            "  Samples will be printed as token IDs."
        )

        return None

    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        print(
            f"[TOKENIZER] Loaded: "
            f"{tokenizer_path.resolve()}"
        )

        return tokenizer

    except Exception as error:
        print(
            f"[TOKENIZER] Could not load tokenizer: {error}\n"
            "  Samples will be printed as token IDs."
        )

        return None


def decode_tokens(tokenizer, tokens):
    if tokenizer is None:
        return " ".join(map(str, tokens.tolist()))

    try:
        return tokenizer.decode(
            tokens.tolist(),
            skip_special_tokens=False,
        )

    except Exception as error:
        return f"[Decode error: {error}]"


# ============================================================
# Model diagnostics
# ============================================================

def print_model_summary(model, config):
    parameters = list(model.parameters())

    total_params = sum(
        parameter.numel()
        for parameter in parameters
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in parameters
        if parameter.requires_grad
    )

    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in parameters
    )

    buffer_bytes = sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
    )

    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)

    print(f"Model class: {model.__class__.__name__}")
    print(f"Parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter memory: {parameter_bytes / GIB:.4f} GiB")
    print(f"Buffer memory: {buffer_bytes / GIB:.4f} GiB")

    print(
        f"Architecture: "
        f"{config.n_layer} layers | "
        f"{config.n_embd} embedding | "
        f"{config.n_head} attention heads"
    )

    print(
        f"MoE: {config.n_experts} experts | "
        f"Top-{config.n_experts_per_token}"
    )

    mtp_heads = getattr(config, "mtp_num_heads", 0)

    print(
        f"MTP: {'ENABLED' if mtp_heads > 0 else 'DISABLED'}"
        f" | Heads: {mtp_heads}"
    )

    print("=" * 80 + "\n")


def print_module_summary(model):
    print("\n" + "=" * 80)
    print("MODULE TREE")
    print("=" * 80)

    for name, module in model.named_modules():
        if not name:
            continue

        params = sum(
            parameter.numel()
            for parameter in module.parameters(recurse=False)
        )

        buffers = sum(
            buffer.numel()
            for buffer in module.buffers(recurse=False)
        )

        if params == 0 and buffers == 0:
            continue

        print(
            f"{name:<65} "
            f"{module.__class__.__name__:<24} "
            f"params={params:,}"
        )

    print("=" * 80 + "\n")


@torch.inference_mode()
def run_component_diagnostics(model, config, device, autocast):
    """
    Executes one small forward pass before training.

    Reports module names, execution time, output shape,
    and CUDA memory snapshots.

    This is a diagnostic pass, not a training iteration.
    """

    print("\n" + "=" * 80)
    print("PRE-TRAINING COMPONENT DIAGNOSTICS")
    print("=" * 80)

    model.eval()

    handles = []
    module_stats = {}

    def make_hook(name):
        def hook(module, inputs, output):
            if device == "cuda":
                torch.cuda.synchronize()

            now = time.perf_counter()

            previous = module_stats.get(name)

            start_time = (
                previous["start"]
                if previous is not None
                else now
            )

            output_shape = None

            if torch.is_tensor(output):
                output_shape = tuple(output.shape)

            elif isinstance(output, (tuple, list)):
                output_shape = [
                    tuple(item.shape)
                    for item in output
                    if torch.is_tensor(item)
                ]

            elif isinstance(output, dict):
                output_shape = {
                    key: tuple(value.shape)
                    for key, value in output.items()
                    if torch.is_tensor(value)
                }

            memory = get_vram()

            module_stats[name] = {
                "start": start_time,
                "time": now - start_time,
                "shape": output_shape,
                "allocated": (
                    memory["allocated"]
                    if memory is not None
                    else 0
                ),
                "peak": (
                    memory["peak_allocated"]
                    if memory is not None
                    else 0
                ),
            }

        return hook

    for name, module in model.named_modules():
        if not name:
            continue

        handles.append(
            module.register_forward_hook(
                make_hook(name)
            )
        )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    try:
        x = torch.randint(
            0,
            config.vocab_size,
            (1, config.block_size),
            device=device,
        )

        y = torch.randint(
            0,
            config.vocab_size,
            (1, config.block_size),
            device=device,
        )

        with autocast():
            output = model(x, y)

        _, loss, _, _ = get_model_loss(output)

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time

        print(f"Diagnostic forward completed in {elapsed:.3f}s")
        print(f"Diagnostic loss: {tensor_to_float(loss):.6f}")

    except Exception as error:
        print(
            "\n[DIAGNOSTIC WARNING] "
            f"Forward diagnostic failed: {error}"
        )

        print(
            "The actual model interface or a diagnostic "
            "configuration may need adjustment."
        )

        raise

    finally:
        for handle in handles:
            handle.remove()

        model.train()

    print("\nCOMPONENT REPORT")

    for name, stats in module_stats.items():
        print(
            f"\n{name}\n"
            f"  Output shape: {stats['shape']}\n"
            f"  Execution time: {stats['time']:.6f}s"
        )

        if device == "cuda":
            print(
                f"  Allocated after component: "
                f"{stats['allocated'] / GIB:.3f} GiB\n"
                f"  Peak allocated: "
                f"{stats['peak'] / GIB:.3f} GiB"
            )

    print("\n" + "=" * 80)
    print("END COMPONENT DIAGNOSTICS")
    print("=" * 80 + "\n")


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def estimate_loss(
    model,
    config,
    train_dataset,
    val_dataset,
    device,
    autocast,
):
    was_training = model.training
    model.eval()

    results = {}

    for split, dataset in (
        ("train", train_dataset),
        ("val", val_dataset),
    ):
        total_losses = []
        main_losses = []
        mtp_losses = {}

        for _ in range(config.eval_iters):
            x, y = dataset.get_batch(config.batch_size)

            with autocast():
                output = model(x, y)

            _, total, main, mtp = get_model_loss(output)

            total_losses.append(
                tensor_to_float(total)
            )

            main_losses.append(
                tensor_to_float(main)
            )

            for name, value in normalize_mtp_losses(mtp).items():
                mtp_losses.setdefault(name, [])

                mtp_losses[name].append(
                    tensor_to_float(value)
                )

        results[split] = {
            "total": float(np.mean(total_losses)),
            "main": float(np.mean(main_losses)),
            "mtp": {
                name: float(np.mean(values))
                for name, values in mtp_losses.items()
            },
        }

    model.train(was_training)

    return results


def print_samples(
    train_dataset,
    val_dataset,
    tokenizer,
    config,
):
    count = int(config.train_samples)

    if count <= 0:
        print("[SAMPLES] train_samples <= 0. Skipping.")
        return

    train_count = count // 2
    val_count = count - train_count

    max_tokens = int(config.train_sample_max_tokens)

    print("\n" + "=" * 80)
    print(
        f"RANDOM DATASET SAMPLES | "
        f"{train_count} TRAIN + {val_count} VALIDATION"
    )
    print("=" * 80)

    for name, dataset, sample_count in (
        ("TRAIN.BIN", train_dataset, train_count),
        ("VAL.BIN", val_dataset, val_count),
    ):
        print(f"\n--- {name} ---")

        snippets = dataset.get_random_snippets(
            sample_count,
            max_tokens,
        )

        for index, snippet in enumerate(snippets, 1):
            text = decode_tokens(tokenizer, snippet)

            print(
                f"\n[{name} SAMPLE {index}]\n"
                f"Tokens: {len(snippet)}\n"
                f"{text}"
            )

    print("\n" + "=" * 80 + "\n")


# ============================================================
# Training
# ============================================================

def train_model(config=None):

    # Explicit configuration injection.
    # No alternate configuration is constructed when supplied.
    if config is None:
        config = ModelConfig()

    print("\n" + "=" * 80)
    print(f"{config.model_family_name} — TRAINING")
    print("=" * 80)

    print(f"Config class: {config.__class__.__name__}")
    print(f"Config source: {ModelConfig.__module__}")
    print(f"Seed: {config.seed}")

    seed_everything(config.seed)

    config.model_save_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        config.device
        if config.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if config.dtype not in dtype_map:
        raise ValueError(
            f"Unsupported dtype: {config.dtype}"
        )

    dtype = dtype_map[config.dtype]

    amp = (
        device == "cuda"
        and dtype != torch.float32
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp and dtype == torch.float16,
    )

    def autocast():
        return torch.autocast(
            device_type=device,
            dtype=dtype,
            enabled=amp,
        )

    print(f"\nDevice: {device}")
    print(f"Dtype: {config.dtype}")
    print(f"AMP enabled: {amp}")
    print(f"Compile enabled: {config.compile}")

    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)

        print(
            f"GPU: {properties.name}\n"
            f"Compute capability: "
            f"{properties.major}.{properties.minor}\n"
            f"Total VRAM: {properties.total_memory / GIB:.3f} GiB"
        )

    print("\nLoading datasets...")

    train_dataset = TokenDataset(
        config.train_path,
        config.block_size,
        device,
        "TRAIN",
    )

    val_dataset = TokenDataset(
        config.val_path,
        config.block_size,
        device,
        "VALIDATION",
    )

    tokenizer = load_tokenizer(config)

    print("\nBuilding model...")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()

    # This is the exact config object passed to Model.
    model = Model(config).to(device)

    if device == "cuda":
        torch.cuda.synchronize()

    model_load_time = time.perf_counter() - load_start

    print(
        f"Model constructed in "
        f"{model_load_time:.3f}s"
    )

    print_vram("After model construction")

    print_model_summary(model, config)
    print_module_summary(model)

    # Diagnostic forward before compilation and training.
    run_component_diagnostics(
        model,
        config,
        device,
        autocast,
    )

    if config.compile:
        print("\nCompiling model...")

        compile_start = time.perf_counter()

        model = torch.compile(model)

        print(
            f"torch.compile wrapper created in "
            f"{time.perf_counter() - compile_start:.3f}s"
        )

        print(
            "Note: actual compilation may happen during "
            "the first training forward."
        )

    raw_model = (
        model._orig_mod
        if hasattr(model, "_orig_mod")
        else model
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
        eps=config.eps,
    )

    total_params = sum(
        parameter.numel()
        for parameter in raw_model.parameters()
    )

    tokens_per_step = (
        config.batch_size
        * config.gradient_accumulation_steps
        * config.block_size
    )

    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)

    print(f"Parameters: {total_params:,}")
    print(f"Batch size: {config.batch_size}")
    print(
        f"Gradient accumulation: "
        f"{config.gradient_accumulation_steps}"
    )

    print(f"Tokens per optimizer step: {tokens_per_step:,}")
    print(f"Maximum iterations: {config.max_iters}")
    print(f"Warmup steps: {config.warmup_steps}")
    print(f"Evaluation interval: {config.eval_interval}")
    print(f"Evaluation batches per split: {config.eval_iters}")

    print(
        f"Training samples per evaluation: "
        f"{config.train_samples}"
    )

    print(
        f"Maximum sample length: "
        f"{config.train_sample_max_tokens}"
    )

    print("=" * 80 + "\n")

    best_val_loss = float("inf")

    optimizer.zero_grad(set_to_none=True)
    model.train()

    training_start = time.perf_counter()

    iteration_times = []
    running_tokens = 0

    progress = tqdm(
        range(1, config.max_iters + 1),
        desc="Training",
        dynamic_ncols=True,
    )

    for step in progress:

        iteration_start = time.perf_counter()

        lr = get_lr(config, step)

        for group in optimizer.param_groups:
            group["lr"] = lr

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        total_loss = 0.0
        main_loss = 0.0
        mtp_loss = {}

        for accumulation_step in range(
            config.gradient_accumulation_steps
        ):
            x, y = train_dataset.get_batch(
                config.batch_size
            )

            with autocast():
                output = model(x, y)

                _, loss, main, mtp = get_model_loss(output)

                loss = (
                    loss
                    / config.gradient_accumulation_steps
                )

            scaler.scale(loss).backward()

            total_loss += tensor_to_float(loss) * (
                config.gradient_accumulation_steps
            )

            main_loss += tensor_to_float(main)

            for name, value in normalize_mtp_losses(mtp).items():
                mtp_loss.setdefault(name, 0.0)

                mtp_loss[name] += tensor_to_float(value)

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        total_loss /= config.gradient_accumulation_steps
        main_loss /= config.gradient_accumulation_steps

        for name in mtp_loss:
            mtp_loss[name] /= (
                config.gradient_accumulation_steps
            )

        if device == "cuda":
            torch.cuda.synchronize()

        iteration_time = (
            time.perf_counter() - iteration_start
        )

        iteration_times.append(iteration_time)

        running_tokens += tokens_per_step

        tokens_per_second = (
            tokens_per_step / max(iteration_time, 1e-9)
        )

        elapsed = (
            time.perf_counter() - training_start
        )

        average_iteration_time = float(
            np.mean(iteration_times)
        )

        remaining_iterations = (
            config.max_iters - step
        )

        eta = (
            average_iteration_time
            * remaining_iterations
        )

        progress.set_postfix(
            loss=f"{total_loss:.4f}",
            main=f"{main_loss:.4f}",
            tok_s=f"{tokens_per_second:,.0f}",
            lr=f"{lr:.2e}",
            eta=format_time(eta),
        )

        # Detailed periodic logging.
        if step % 10 == 0 or step == 1:
            log = (
                f"\n{'=' * 80}\n"
                f"STEP {step:,}/{config.max_iters:,}\n"
                f"{'=' * 80}\n"
                f"Total loss: {total_loss:.6f}\n"
                f"Main loss:  {main_loss:.6f}\n"
            )

            if mtp_loss:
                for name, value in mtp_loss.items():
                    log += (
                        f"{name} loss: "
                        f"{value:.6f}\n"
                    )
            else:
                log += "MTP: DISABLED\n"

            log += (
                f"Learning rate: {lr:.8e}\n"
                f"Iteration time: {iteration_time:.4f}s\n"
                f"Average iteration: "
                f"{average_iteration_time:.4f}s\n"
                f"Tokens/sec: {tokens_per_second:,.2f}\n"
                f"Total tokens: {running_tokens:,}\n"
                f"Elapsed: {format_time(elapsed)}\n"
                f"ETA: {format_time(eta)}\n"
            )

            tqdm.write(log)

            print_vram(f"Step {step}")

        # ====================================================
        # Evaluation
        # ====================================================

        should_evaluate = (
            step % config.eval_interval == 0
            or step == config.max_iters
        )

        if should_evaluate:

            print("\n" + "=" * 80)
            print(f"EVALUATION AT STEP {step}")
            print("=" * 80)

            evaluation_start = time.perf_counter()

            losses = estimate_loss(
                model,
                config,
                train_dataset,
                val_dataset,
                device,
                autocast,
            )

            evaluation_time = (
                time.perf_counter()
                - evaluation_start
            )

            print(
                f"\nEvaluation completed in "
                f"{evaluation_time:.3f}s"
            )

            print("\nTRAIN")
            print_loss_summary(losses["train"])

            print("\nVALIDATION")
            print_loss_summary(losses["val"])

            print_vram(
                f"After evaluation at step {step}"
            )

            # Best checkpoint.
            if losses["val"]["total"] < best_val_loss:
                best_val_loss = losses["val"]["total"]

                best_path = (
                    config.model_save_folder
                    / "best_model.pt"
                )

                torch.save(
                    raw_model.state_dict(),
                    best_path,
                )

                print(
                    f"\n[BEST CHECKPOINT]\n"
                    f"Path: {best_path.resolve()}\n"
                    f"Validation loss: "
                    f"{best_val_loss:.6f}"
                )

            # Latest checkpoint.
            latest_path = (
                config.model_save_folder
                / "latest_model.pt"
            )

            torch.save(
                raw_model.state_dict(),
                latest_path,
            )

            print(
                f"\n[LATEST CHECKPOINT]\n"
                f"Path: {latest_path.resolve()}"
            )

            # Print random train/validation samples.
            print_samples(
                train_dataset,
                val_dataset,
                tokenizer,
                config,
            )

    total_training_time = (
        time.perf_counter()
        - training_start
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(f"Total training time: {format_time(total_training_time)}")

    print(
        f"Average iteration time: "
        f"{np.mean(iteration_times):.4f}s"
    )

    print(
        f"Average tokens/sec: "
        f"{tokens_per_step / np.mean(iteration_times):,.2f}"
    )

    print(f"Total optimizer steps: {config.max_iters:,}")

    print(f"Total tokens processed: {running_tokens:,}")

    print(f"Best validation loss: {best_val_loss:.6f}")

    print_vram("Final VRAM")

    print("=" * 80)

    return model, config


if __name__ == "__main__":
    train_model()
