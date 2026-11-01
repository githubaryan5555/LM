
import math
import numpy as np
import torch
from tqdm.auto import tqdm
from model import Model
from config import ModelConfig


def train_model():
    config = ModelConfig()
    config.model_save_folder.mkdir(parents=True, exist_ok=True)

    device = config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[config.dtype]
    amp = device == "cuda" and dtype != torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)

    def autocast():
        return torch.autocast(device_type=device, dtype=dtype, enabled=amp)

    # Pre-tokenized uint16 datasets
    train_data = np.memmap(config.train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(config.val_path, dtype=np.uint16, mode="r")

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = np.random.randint(len(data) - config.block_size - 1, size=config.batch_size)
        x = np.stack([data[i:i + config.block_size] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + config.block_size + 1] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    # Model and compilation
    torch.manual_seed(42)
    model = Model(config).to(device)
    if getattr(config, "compile", False):
        print("Compiling model...")
        model = torch.compile(model)
        print("Compile enabled.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas,
                                  weight_decay=config.weight_decay, eps=config.eps)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    params = sum(p.numel() for p in raw_model.parameters())

    print(f"Parameters: {params:,}")
    print(f"Device: {device} | dtype: {config.dtype} | compile: {getattr(config, 'compile', False)}")
    print(f"Batch: {config.batch_size} | Accumulation: {config.gradient_accumulation_steps}")
    print(f"Tokens/step: {config.batch_size * config.gradient_accumulation_steps * config.block_size:,}")
    print(f"Evaluation: every {config.eval_interval} steps, {config.eval_iters} batches per split")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    @torch.inference_mode()
    def estimate_loss():
        model.eval()
        losses = {}
        for split in ("train", "val"):
            total = 0.0
            for _ in range(config.eval_iters):
                X, Y = get_batch(split)
                with autocast():
                    _, loss, _, _ = model(X, Y)
                total += loss.item()
            losses[split] = total / config.eval_iters
        model.train()
        return losses

    best_val_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)
    model.train()

    for step in tqdm(range(1, config.max_iters + 1)):
        if step <= config.warmup_steps:
            lr = config.learning_rate * step / config.warmup_steps
        else:
            progress = (step - config.warmup_steps) / max(1, config.max_iters - config.warmup_steps)
            lr = config.min_lr + (config.learning_rate - config.min_lr) * (1 + math.cos(math.pi * progress)) / 2

        for group in optimizer.param_groups:
            group["lr"] = lr

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        total_loss = main_loss = mtp_loss = 0.0

        for _ in range(config.gradient_accumulation_steps):
            X, Y = get_batch("train")
            with autocast():
                _, loss, main, mtp = model(X, Y)
                loss = loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            total_loss += loss.detach().item()
            main_loss += main.detach().item()
            mtp_loss += mtp.detach().item()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss /= config.gradient_accumulation_steps
        main_loss /= config.gradient_accumulation_steps
        mtp_loss /= config.gradient_accumulation_steps

        if step % 10 == 0:
            log = f"Step {step} | Loss {total_loss:.4f} | Main {main_loss:.4f} | MTP {mtp_loss:.4f} | LR {lr:.2e}"
            if device == "cuda":
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                peak = torch.cuda.max_memory_allocated() / 1024**3
                free, total = torch.cuda.mem_get_info()
                log += f" | VRAM allocated {allocated:.2f}GB | reserved {reserved:.2f}GB | peak {peak:.2f}GB | free {free / 1024**3:.2f}GB"
            tqdm.write(log)

        if step % config.eval_interval == 0 or step == config.max_iters:
            losses = estimate_loss()
            print(f"Evaluation {step} | Train {losses['train']:.4f} | Val {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(raw_model.state_dict(), config.model_save_folder / "best_model.pt")
                print(f"Saved best model | Val loss: {best_val_loss:.4f}")

    print(f"Training complete | Best validation loss: {best_val_loss:.4f}")
    return model, config


if __name__ == "__main__":
    train_model()
