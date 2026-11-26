"""Train Robbie-300M (~304.8M parameters) Chatbot with strict <= 5.0 GB VRAM guard.

Architecture:
- n_layer: 24
- n_head: 16 (head_dim = 64)
- n_embd: 1024
- block_size: 512
- vocab_size: 2048 (BPETokenizer)
- Total parameters: 304,834,560 (~304.8 Million parameters)

Hardware Safety Safeguards:
- Hard CUDA memory cap at 4.8 GB (strictly under 5.0 GB limit, leaving >3.0 GB untouched for OS)
- 8-bit AdamW optimizer via bitsandbytes (~600 MB optimizer states)
- Automatic Mixed Precision (AMP FP16) on NVIDIA Tensor Cores
- Thermal pacing cooldown pauses (12-second sleep every 100 steps)
"""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except Exception:
    bnb = None
    HAS_BNB = False

from robbie.core.data.chat_data import collate_answers, load_prepared
from robbie.core.models.transformer import RobbieTransformer, TransformerConfig
from robbie.core.tokenizer.bpe_tokenizer import BPETokenizer
from robbie.core.training.chat_profiles import basic_samples


class SafeVRAMGuard:
    """Enforces strict <= 5.0 GB VRAM limit and thermal pacing."""

    def __init__(self, limit_gb: float = 4.8, cooldown_every: int = 100, cooldown_sec: float = 12.0, target_gpu: int = 0):
        self.limit_gb = limit_gb
        self.cooldown_every = cooldown_every
        self.cooldown_sec = cooldown_sec
        self.target_gpu = target_gpu
        self.cooldown_count = 0

        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(target_gpu).total_memory / (1024 ** 3)
            # Enforce hard ceiling: 4.8 GB max
            fraction = min(0.60, limit_gb / total_vram)
            try:
                torch.cuda.set_per_process_memory_fraction(fraction, target_gpu)
                print(f"[VRAM Guard] Enforced hard allocator ceiling: {fraction*total_vram:.2f} GB ({fraction*100:.1f}% of {total_vram:.1f}GB)", flush=True)
            except Exception as e:
                print(f"[VRAM Guard] Memory fraction notice: {e}", flush=True)

    def check(self, step: int):
        if not torch.cuda.is_available():
            return

        allocated_gb = torch.cuda.memory_allocated(self.target_gpu) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(self.target_gpu) / (1024 ** 3)

        # Scheduled Thermal Pacing Cooldown
        if step > 0 and step % self.cooldown_every == 0:
            torch.cuda.empty_cache()
            self.cooldown_count += 1
            print(f"[VRAM Guard] Thermal pacing cooldown at step {step}: VRAM={reserved_gb:.2f}GB. Cooling down GPU for {self.cooldown_sec}s...", flush=True)
            time.sleep(self.cooldown_sec)
            torch.cuda.empty_cache()

        # Emergency cleanup if approaching limit
        if reserved_gb > (self.limit_gb - 0.3):
            torch.cuda.empty_cache()
            time.sleep(1.0)


def get_cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def train_300m_model(
    model: RobbieTransformer,
    train_samples: list,
    val_samples: list,
    tokenizer: BPETokenizer,
    epochs: int,
    batch_size: int,
    accumulation_steps: int,
    base_lr: float,
    min_lr: float,
    output_dir: Path,
    device: torch.device,
    vram_guard: SafeVRAMGuard,
    eval_every: int = 200,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.train()

    # 8-bit AdamW optimizer saves >2.5 GB of VRAM when available
    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if HAS_BNB and bnb is not None:
        optimizer = bnb.optim.AdamW8bit(param_groups, lr=base_lr)
    else:
        print("[Notice] bitsandbytes not available; falling back to torch.optim.AdamW.", flush=True)
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    effective_batch = batch_size * accumulation_steps
    steps_per_epoch = math.ceil(len(train_samples) / effective_batch)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(30, int(total_steps * 0.04))

    print(f"\n{'='*70}")
    print(f"  STARTING ROBBIE-300M CONVERSATIONAL TRAINING")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs: {epochs} | Steps/Epoch: {steps_per_epoch} | Total Updates: {total_steps}")
    print(f"  Batch: {batch_size} (Micro) x {accumulation_steps} (Acc) = {effective_batch} (Effective)")
    print(f"  Base LR: {base_lr} -> {min_lr} (Warmup: {warmup_steps} steps)")
    print(f"  Hardware Guard: Max VRAM <= 4.8 GB | Thermal Pacing enabled")
    print(f"{'='*70}\n", flush=True)

    best_val_loss = float("inf")
    global_step = 0
    t0 = time.time()

    for epoch in range(epochs):
        random.shuffle(train_samples)
        cursor = 0
        optimizer.zero_grad()

        while cursor < len(train_samples):
            accum_loss = 0.0
            for micro_step in range(accumulation_steps):
                if cursor >= len(train_samples):
                    break
                batch = train_samples[cursor : cursor + batch_size]
                cursor += len(batch)
                x, y = collate_answers(batch, tokenizer.pad_token_id)
                x, y = x.to(device), y.to(device)

                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    _, loss = model(x, targets=y)
                    loss = loss / accumulation_steps

                accum_loss += loss.item()
                scaler.scale(loss).backward()

            # Optimizer update
            lr = get_cosine_lr(global_step, total_steps, base_lr, min_lr, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()
            global_step += 1

            # Monitor VRAM and thermal pacing
            vram_guard.check(global_step)

            if global_step % 20 == 0 or global_step == total_steps:
                elapsed = time.time() - t0
                speed = global_step / max(1.0, elapsed)
                vram_gb = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                vram_res = torch.cuda.memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                print(
                    f"Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | "
                    f"Loss: {accum_loss:.4f} | LR: {lr:.2e} | "
                    f"Speed: {speed:.2f} step/s | VRAM: {vram_gb:.2f}GB (alloc) / {vram_res:.2f}GB (res)",
                    flush=True,
                )

            # Periodic Validation
            if global_step % eval_every == 0 or global_step == total_steps:
                val_loss = evaluate_loss(model, val_samples, tokenizer, batch_size, device)
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    best_path = output_dir / "best_model.pt"
                    save_payload = {
                        "model_state_dict": model.state_dict(),
                        "config": model.config,
                        "epoch": epoch,
                        "step": global_step,
                        "val_loss": val_loss,
                        "val_perplexity": math.exp(min(val_loss, 20.0)),
                        "dialogue_format": "assistant-eos-v1",
                    }
                    torch.save(save_payload, best_path)
                    print(f"  --> [BEST CHECKPOINT] Saved model (val_loss: {val_loss:.4f}, ppl: {math.exp(min(val_loss, 20.0)):.2f})", flush=True)
                else:
                    print(f"  [Validation] val_loss: {val_loss:.4f} (best: {best_val_loss:.4f})", flush=True)
                model.train()

        # End of epoch cooldown pause
        print(f"\n--- Epoch {epoch+1} Complete. Performing 15-second cooldown cycle ---", flush=True)
        torch.cuda.empty_cache()
        time.sleep(15.0)

    print(f"\nRobbie-300M training finished in {time.time()-t0:.1f}s. Lowest val loss: {best_val_loss:.4f}\n", flush=True)
    return best_val_loss


@torch.no_grad()
def evaluate_loss(model, samples, tokenizer, batch_size, device, max_eval=300):
    was_training = model.training
    model.eval()
    samples = samples[:max_eval]
    total_loss, total_tokens = 0.0, 0
    try:
        for i in range(0, len(samples), batch_size):
            chunk = samples[i : i + batch_size]
            x, y = collate_answers(chunk, tokenizer.pad_token_id)
            x, y = x.to(device), y.to(device)
            valid_tokens = int((y != -100).sum())
            if valid_tokens == 0:
                continue
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                _, loss = model(x, targets=y)
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
        return total_loss / max(1, total_tokens)
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/chat_300m_v1/answers")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/chat_300m_v1")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.8e-4)
    parser.add_argument("--min-lr", type=float, default=1.8e-5)
    parser.add_argument("--vram-limit-gb", type=float, default=4.8)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--resume", type=Path, default=None, help="Resume or continue fine-tuning from an existing checkpoint")
    args = parser.parse_args()

    vram_guard = SafeVRAMGuard(limit_gb=args.vram_limit_gb, cooldown_every=100, cooldown_sec=12.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    tokenizer_path = ROOT / "checkpoints/bpe_stage1_vocab.json"
    tokenizer = BPETokenizer.load(tokenizer_path)

    # Robbie-300M Configuration
    # 24 layers, 16 heads, 1024 hidden dimension, 512 context window = ~304.8 Million parameters
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=512,
        n_layer=24,
        n_head=16,
        n_embd=1024,
        dropout=0.05,
    )

    model = RobbieTransformer(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nRobbie-300M Model Instantiated:")
    print(f"  Layers: {config.n_layer} | Heads: {config.n_head} | Embedding: {config.n_embd}")
    print(f"  Context Window: {config.block_size} tokens | Vocab: {config.vocab_size} tokens")
    print(f"  Total Parameters: {param_count:,} (~{param_count / 1e6:.2f} Million parameters)\n", flush=True)

    if args.resume:
        resume_file = Path(args.resume)
        if not resume_file.is_absolute():
            resume_file = ROOT / resume_file
        if resume_file.is_file():
            print(f"[Checkpoint] Loading existing weights from: {resume_file}", flush=True)
            loaded_payload = torch.load(resume_file, map_location="cpu", weights_only=False)
            sd = loaded_payload.get("model_state_dict", loaded_payload)
            model.load_state_dict(sd)
            print("  --> Weights restored successfully! Continuing training...", flush=True)
        else:
            print(f"[Warning] Specified resume file {resume_file} not found; initializing from scratch.", flush=True)

    # Load prepared data
    splits, manifest = load_prepared(args.data_dir, tokenizer_path)
    print(f"Loaded {len(splits['train']):,} training samples and {len(splits['validation']):,} validation samples", flush=True)

    stage2_ckpt_path = ROOT / "checkpoints/stage2_300m_conversational_transformer.pt"

    # Train model
    best_loss = train_300m_model(
        model=model,
        train_samples=splits["train"],
        val_samples=splits["validation"],
        tokenizer=tokenizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        base_lr=args.lr,
        min_lr=args.min_lr,
        output_dir=args.output_dir,
        device=device,
        vram_guard=vram_guard,
        eval_every=args.eval_every,
    )

    # Save canonical 300M checkpoint
    best_ckpt = torch.load(args.output_dir / "best_model.pt", map_location="cpu", weights_only=False)
    torch.save(best_ckpt, stage2_ckpt_path)
    print(f"\nSaved canonical Robbie-300M checkpoint to: {stage2_ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
