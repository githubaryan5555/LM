
import math
import numpy as np
import torch
import torch_tpu
from tqdm.auto import tqdm
from model import Model
from config import ModelConfig


def train_model():
    config = ModelConfig()
    config.model_save_folder.mkdir(parents=True, exist_ok=True)

    # TPU device
    device = torch.device("tpu")

    # Precision
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }[config.dtype]

    # BF16 recommended for TPU
    amp = dtype != torch.float32

    def autocast():
        return torch.autocast(
            device_type="tpu",
            dtype=dtype,
            enabled=amp
        )

    # Pre-tokenized uint16 datasets
    train_data = np.memmap(
        config.train_path,
        dtype=np.uint16,
        mode="r"
    )

    val_data = np.memmap(
        config.val_path,
        dtype=np.uint16,
        mode="r"
    )

    def get_batch(split):
        data = train_data if split == "train" else val_data

        ix = np.random.randint(
            len(data) - config.block_size - 1,
            size=config.batch_size
        )

        x = np.stack([
            data[i:i + config.block_size] for i in ix
        ]).astype(np.int64)

        y = np.stack([
            data[i + 1:i + config.block_size + 1] for i in ix
        ]).astype(np.int64)

        return (
            torch.from_numpy(x).to(device),
            torch.from_numpy(y).to(device)
        )

    # Model
    torch.manual_seed(42)

    model = Model(config).to(device)

    if getattr(config, "compile", False):
        print("Compiling model...")
        model = torch.compile(model)
        print("Compile enabled.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
        eps=config.eps
    )

    raw_model = (
        model._orig_mod
        if hasattr(model, "_orig_mod")
        else model
    )

    params = sum(
        p.numel() for p in raw_model.parameters()
    )

    print(f"Parameters: {params:,}")
    print(f"Device: {device} | dtype: {config.dtype} | compile: {getattr(config, 'compile', False)}")
    print(f"Batch: {config.batch_size} | Accumulation: {config.gradient_accumulation_steps}")
    print(
        f"Tokens/step: "
        f"{config.batch_size * config.gradient_accumulation_steps * config.block_size:,}"
    )
    print(
        f"Evaluation: every {config.eval_interval} steps, "
        f"{config.eval_iters} batches per split"
    )

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

        # Learning rate schedule
        if step <= config.warmup_steps:
            lr = (
                config.learning_rate
                * step
                / config.warmup_steps
            )

        else:
            progress = (
                (step - config.warmup_steps)
                / max(
                    1,
                    config.max_iters - config.warmup_steps
                )
            )

            lr = config.min_lr + (
                config.learning_rate - config.min_lr
            ) * (
                1 + math.cos(math.pi * progress)
            ) / 2

        for group in optimizer.param_groups:
            group["lr"] = lr

        total_loss = 0.0
        main_loss = 0.0
        mtp_loss = 0.0

        # Gradient accumulation
        for _ in range(config.gradient_accumulation_steps):

            X, Y = get_batch("train")

            with autocast():
                _, loss, main, mtp = model(X, Y)

                loss = (
                    loss /
                    config.gradient_accumulation_steps
                )

            loss.backward()

            total_loss += loss.detach().item()
            main_loss += main.detach().item()

            if mtp is not None:
                mtp_loss += mtp.detach().item()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss /= config.gradient_accumulation_steps
        main_loss /= config.gradient_accumulation_steps
        mtp_loss /= config.gradient_accumulation_steps

        # Logging
        if step % 10 == 0:

            log = (
                f"Step {step} | "
                f"Loss {total_loss:.4f} | "
                f"Main {main_loss:.4f} | "
                f"MTP {mtp_loss:.4f} | "
                f"LR {lr:.2e}"
            )

            tqdm.write(log)

        # Evaluation
        if (
            step % config.eval_interval == 0
            or step == config.max_iters
        ):

            losses = estimate_loss()

            print(
                f"Evaluation {step} | "
                f"Train {losses['train']:.4f} | "
                f"Val {losses['val']:.4f}"
            )

            if losses["val"] < best_val_loss:

                best_val_loss = losses["val"]

                # Save best model
                cpu_state_dict = {
                    k: v.detach().cpu()
                    for k, v in raw_model.state_dict().items()
                }

                torch.save(
                    cpu_state_dict,
                    config.model_save_folder / "best_model.pt"
                )

                print(
                    f"Saved best model | "
                    f"Val loss: {best_val_loss:.4f}"
                )

    print(
        f"Training complete | "
        f"Best validation loss: {best_val_loss:.4f}"
    )

    return model, config


if __name__ == "__main__":
    train_model()
