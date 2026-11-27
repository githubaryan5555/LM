
import math
import time
import random
import numpy as np
import torch
from tqdm.auto import tqdm
from model import Model
from config import ModelConfig


def train_model():
    config = ModelConfig()
    config.model_save_folder.mkdir(parents=True, exist_ok=True)

    # Seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Device and dtype
    device = config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[config.dtype]
    amp = device == "cuda" and dtype != torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)

    def autocast():
        return torch.autocast(device_type=device, dtype=dtype, enabled=amp)

    # Datasets
    train_data = np.memmap(config.train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(config.val_path, dtype=np.uint16, mode="r")

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = np.random.randint(len(data) - config.block_size - 1,
                               size=config.batch_size)
        x = np.stack([data[i:i + config.block_size] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + config.block_size + 1] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    # Tokenizer (for sampling)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(config.tokenizer_path))

    # Model
    model = Model(config).to(device)
    raw_model = model

    if config.compile:
        print("Compiling model...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
        eps=config.eps
    )

    params = sum(p.numel() for p in raw_model.parameters())
    tokens_per_step = (
        config.batch_size * config.gradient_accumulation_steps * config.block_size
    )

    print(f"Parameters: {params:,}")
    print(f"Device: {device} | dtype: {config.dtype} | Compile: {config.compile}")
    print(f"Batch: {config.batch_size} | Accumulation: {config.gradient_accumulation_steps}")
    print(f"Tokens/step: {tokens_per_step:,}")
    print(f"Train/Val tokens: {len(train_data):,} / {len(val_data):,}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Evaluation
    @torch.inference_mode()
    def estimate_loss():
        model.eval()
        results = {}

        for split in ("train", "val"):
            total = 0.0
            for _ in range(config.eval_iters):
                X, Y = get_batch(split)
                with autocast():
                    _, loss, _, _ = model(X, Y)
                total += loss.item()
            results[split] = total / config.eval_iters

        model.train()
        return results

    # Text samples
    @torch.inference_mode()
    def sample_text(split, count=None):
        data = train_data if split == "train" else val_data
        count = count or config.train_samples
        model.eval()
        samples = []

        for _ in range(count):
            length = min(config.train_sample_max_tokens, len(data) - 1)
            start = np.random.randint(0, len(data) - length)
            ids = torch.tensor(
                data[start:start + min(16, length)].astype(np.int64),
                device=device
            ).unsqueeze(0)

            for _ in range(config.sample_size):
                idx = ids[:, -config.block_size:]
                with autocast():
                    output = model(idx)

                logits = output[0] if isinstance(output, tuple) else output
                logits = logits[:, -1, :].float()
                logits = logits / config.temperature

                if config.top_k > 0:
                    k = min(config.top_k, logits.size(-1))
                    threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits[logits < threshold] = float("-inf")

                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1), 1
                )
                ids = torch.cat((ids, next_token), dim=1)

            samples.append(tokenizer.decode(ids[0].tolist()))

        model.train()
        return samples

    # Training
    best_val_loss = float("inf")
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    model.train()

    for step in tqdm(range(1, config.max_iters + 1)):

        # LR schedule
        if step <= config.warmup_steps:
            lr = config.learning_rate * step / max(1, config.warmup_steps)
        else:
            progress = (step - config.warmup_steps) / max(
                1, config.max_iters - config.warmup_steps
            )
            progress = min(progress, 1.0)
            lr = config.min_lr + (
                config.learning_rate - config.min_lr
            ) * (1 + math.cos(math.pi * progress)) / 2

        for group in optimizer.param_groups:
            group["lr"] = lr

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        total_loss = main_loss = mtp_loss = 0.0
        step_start = time.time()

        # Gradient accumulation
        for _ in range(config.gradient_accumulation_steps):
            X, Y = get_batch("train")

            with autocast():
                _, loss, main, mtp = model(X, Y)
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            total_loss += loss.detach().item()
            main_loss += main.detach().item()
            if mtp is not None:
                mtp_loss += mtp.detach().item()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss /= config.gradient_accumulation_steps
        main_loss /= config.gradient_accumulation_steps
        mtp_loss /= config.gradient_accumulation_steps

        # Logging
        if step % config.print_every == 0:
            elapsed = time.time() - step_start
            tok_per_sec = tokens_per_step / max(elapsed, 1e-9)

            log = (
                f"Step {step}/{config.max_iters} | "
                f"Loss {total_loss:.4f} | Main {main_loss:.4f} | "
                f"MTP {mtp_loss:.4f} | LR {lr:.2e} | "
                f"{tok_per_sec:,.0f} tok/s | "
                f"Step {elapsed:.2f}s | "
                f"Total {time.time() - start_time:.1f}s"
            )

            if device == "cuda":
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                peak = torch.cuda.max_memory_allocated() / 1024**3
                free, _ = torch.cuda.mem_get_info()
                log += (
                    f" | VRAM {allocated:.2f}/{reserved:.2f}GB "
                    f"| Peak {peak:.2f}GB | Free {free / 1024**3:.2f}GB"
                )

            tqdm.write(log)

        # Evaluation and checkpointing
        if step % config.eval_interval == 0 or step == config.max_iters:
            losses = estimate_loss()
            print(
                f"\nEvaluation {step} | "
                f"Train {losses['train']:.4f} | Val {losses['val']:.4f}"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    raw_model.state_dict(),
                    config.model_save_folder / "best_model.pt"
                )
                print(f"Saved best model | Val {best_val_loss:.4f}")

            # Random train and validation samples
            for split in ("train", "val"):
                print(f"\n--- {split.upper()} SAMPLES ---")
                for i, sample in enumerate(sample_text(split)):
                    print(f"[{i + 1}] {sample}\n")

    print(f"Training complete | Best validation loss: {best_val_loss:.4f}")
    return raw_model, config


if __name__ == "__main__":
    train_model()
