
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
    device_type = "cuda" if device == "cuda" else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[config.dtype]
    amp = device_type == "cuda"
    ctx = lambda: torch.autocast(device_type=device_type, dtype=dtype, enabled=amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)

    # Data loader: pre-tokenized uint16 bins
    train_data = np.memmap(config.train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(config.val_path, dtype=np.uint16, mode="r")

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - config.block_size - 1, (config.batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + config.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + config.block_size + 1].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = {}
        for split in ("train", "val"):
            total = 0.0
            for _ in range(config.eval_iters):
                X, Y = get_batch(split)
                with ctx():
                    _, loss, _, _ = model(X, Y)
                total += loss.item()
            losses[split] = total / config.eval_iters
        model.train()
        return losses

    # Model and optimizer
    model = Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas,
                                  weight_decay=config.weight_decay, eps=config.eps)
    best_val_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device} | dtype: {config.dtype}")
    print(f"Tokens per optimizer step: {config.batch_size * config.gradient_accumulation_steps * config.block_size:,}")

    model.train()
    for step in tqdm(range(1, config.max_iters + 1)):
        if step <= config.warmup_steps:
            lr = config.learning_rate * step / config.warmup_steps
        else:
            progress = (step - config.warmup_steps) / (config.max_iters - config.warmup_steps)
            lr = config.min_lr + (config.learning_rate - config.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        for group in optimizer.param_groups:
            group["lr"] = lr

        total_loss = main_loss = mtp_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            X, Y = get_batch("train")
            with ctx():
                _, loss, main, mtp = model(X, Y)
                loss = loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            total_loss += loss.detach().item()
            main_loss += main.detach().item()
            mtp_loss += mtp.detach().item()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0:
            tqdm.write(f"Step {step} | loss {total_loss:.4f} | main {main_loss:.4f} | mtp {mtp_loss:.4f} | lr {lr:.2e}")

        if step % 100 == 0 or step == config.max_iters:
            losses = estimate_loss()
            print(f"Step {step} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(model.state_dict(), config.model_save_folder / "best_model.pt")

    print(f"Training complete | Best val loss: {best_val_loss:.4f}")
    return model, config


if __name__ == "__main__":
    train_model()
