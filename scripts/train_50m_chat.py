"""Train Robbie-50M (~51.6M parameters) Chatbot with strict 4GB VRAM safety guard.

Architecture:
- n_layer: 16
- n_head: 8 (head_dim = 64)
- n_embd: 512
- block_size: 384
- vocab_size: 2048 (BPETokenizer)
- Total parameters: 51,651,584 (~51.65 Million parameters)

Curriculum:
Stage 1: Conversational Language Foundation (dialogue_lm)
Stage 2: Assistant Instruction Supervised Fine-Tuning (assistant_sft)
"""

import argparse
from collections import Counter
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

from robbie.core.data.chat_data import collate_answers, digest_file, load_prepared
from robbie.core.models.transformer import RobbieTransformer, TransformerConfig
from robbie.core.tokenizer.bpe_tokenizer import BPETokenizer
from robbie.core.training.generation_quality import assess_samples
from robbie.core.training.chat_profiles import basic_samples


class VRAMGuard:
    """Monitors and enforces strict <= 4.0 GB VRAM limit with automatic slow-down."""

    def __init__(self, limit_gb: float = 4.0, target_gpu: int = 0):
        self.limit_gb = limit_gb
        self.target_gpu = target_gpu
        self.slowdowns = 0
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(target_gpu).total_memory / (1024 ** 3)
            # Set hard upper limit on PyTorch allocator (e.g. 48% of 8GB = ~3.84 GB)
            cap_fraction = min(0.48, (limit_gb - 0.1) / total_vram)
            try:
                torch.cuda.set_per_process_memory_fraction(cap_fraction, target_gpu)
                print(f"[VRAM Guard] Enforced hard allocator ceiling: {cap_fraction*total_vram:.2f} GB ({cap_fraction*100:.1f}% of {total_vram:.1f}GB)", flush=True)
            except Exception as e:
                print(f"[VRAM Guard] Warning setting fraction: {e}", flush=True)

    def check(self, step: int):
        if not torch.cuda.is_available():
            return
        allocated_gb = torch.cuda.memory_allocated(self.target_gpu) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(self.target_gpu) / (1024 ** 3)

        # Slow down if approaching 3.5 GB
        if reserved_gb > 3.5 or allocated_gb > 3.0:
            torch.cuda.empty_cache()
            self.slowdowns += 1
            time.sleep(0.05)  # Slow down execution
            if step % 25 == 0:
                print(f"[VRAM Guard] Throttling step {step}: VRAM reserved={reserved_gb:.2f}GB, allocated={allocated_gb:.2f}GB", flush=True)

        # Emergency cooldown if exceeding limit
        if reserved_gb > self.limit_gb:
            print(f"[VRAM Guard] WARNING: VRAM ({reserved_gb:.2f}GB) exceeded {self.limit_gb}GB! Cooling down...", flush=True)
            torch.cuda.empty_cache()
            time.sleep(0.3)


def get_cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def train_stage(
    stage_name: str,
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
    vram_guard: VRAMGuard,
    eval_every: int = 150,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad and p.ndim >= 2], "weight_decay": 0.01},
        {"params": [p for p in model.parameters() if p.requires_grad and p.ndim < 2], "weight_decay": 0.0},
    ], lr=base_lr)

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not use_bf16)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    effective_batch = batch_size * accumulation_steps
    steps_per_epoch = math.ceil(len(train_samples) / effective_batch)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(20, int(total_steps * 0.05))

    print(f"\n{'='*65}")
    print(f"  STARTING {stage_name.upper()} TRAINING")
    print(f"  Epochs: {epochs} | Steps/Epoch: {steps_per_epoch} | Total Updates: {total_steps}")
    print(f"  Batch: {batch_size} (Micro) x {accumulation_steps} (Acc) = {effective_batch} (Effective)")
    print(f"  LR: {base_lr} -> {min_lr} (Warmup: {warmup_steps} steps) | Precision: {amp_dtype}")
    print(f"{'='*65}\n", flush=True)

    best_val_loss = float("inf")
    global_step = 0
    t0 = time.time()

    for epoch in range(epochs):
        random.shuffle(train_samples)
        cursor = 0
        optimizer.zero_grad()

        while cursor < len(train_samples):
            # Accumulate gradients
            accum_loss = 0.0
            for micro_step in range(accumulation_steps):
                if cursor >= len(train_samples):
                    break
                batch = train_samples[cursor : cursor + batch_size]
                cursor += len(batch)
                x, y = collate_answers(batch, tokenizer.pad_token_id)
                x, y = x.to(device), y.to(device)

                with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    _, loss = model(x, targets=y)
                    loss = loss / accumulation_steps

                accum_loss += loss.item()

                if use_bf16:
                    loss.backward()
                else:
                    scaler.scale(loss).backward()

            # Optimizer update
            lr = get_cosine_lr(global_step, total_steps, base_lr, min_lr, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if use_bf16:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            optimizer.zero_grad()
            global_step += 1

            # Monitor VRAM
            vram_guard.check(global_step)

            if global_step % 20 == 0 or global_step == total_steps:
                elapsed = time.time() - t0
                speed = global_step / max(1.0, elapsed)
                vram_gb = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                print(
                    f"Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | "
                    f"Loss: {accum_loss:.4f} | LR: {lr:.2e} | "
                    f"Speed: {speed:.1f} steps/s | VRAM: {vram_gb:.2f}GB",
                    flush=True,
                )

            # Periodic Validation
            if global_step % eval_every == 0 or global_step == total_steps:
                val_loss = evaluate_loss(model, val_samples, tokenizer, batch_size, device, amp_dtype)
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
                    print(f"  --> Saved BEST model checkpoint (val_loss: {val_loss:.4f}, ppl: {math.exp(min(val_loss, 20.0)):.2f})", flush=True)
                else:
                    print(f"  [Validation] val_loss: {val_loss:.4f} (best: {best_val_loss:.4f})", flush=True)
                model.train()

    print(f"\n{stage_name.upper()} finished in {time.time()-t0:.1f}s. Lowest val loss: {best_val_loss:.4f}\n", flush=True)
    return best_val_loss


@torch.no_grad()
def evaluate_loss(model, samples, tokenizer, batch_size, device, amp_dtype, max_eval=300):
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
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                _, loss = model(x, targets=y)
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
        return total_loss / max(1, total_tokens)
    finally:
        model.train(was_training)


def sample_scenarios(model, tokenizer, device, max_tokens=120):
    model.eval()
    test_questions = [
        "hi",
        "whats the time now",
        "what is python",
        "hello",
        "web",
        "What is the difference between RAM and storage?",
        "Explain that more simply.",
        "I want to learn Python. Where should I begin?",
        "What makes a good friend?",
        "Tell me a joke.",
    ]
    results = []
    print("\n" + "=" * 65)
    print("  EVALUATION GENERATIONS (50M MODEL)")
    print("=" * 65)
    for q in test_questions:
        prompt_text = f"User: {q}\nRobbie:"
        ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        output = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=0.55,
            top_k=40,
            repetition_penalty=1.2,
            stop_token_id=tokenizer.eos_token_id,
        )
        gen_ids = output[0, len(ids):].tolist()
        stopped_on_eos = bool(gen_ids and gen_ids[-1] == tokenizer.eos_token_id)
        if stopped_on_eos:
            gen_ids = gen_ids[:-1]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        # Clean potential tags
        for stop_tag in ["\nUser:", "User:", "\nRobbie:", "Robbie:", "<|endoftext|>"]:
            if stop_tag in answer:
                answer = answer.split(stop_tag)[0].strip()

        print(f"\nUser: {q}")
        print(f"Robbie: {answer}")
        print(f"  [Tokens: {len(gen_ids)} | Stopped on EOS: {stopped_on_eos}]")
        results.append({
            "question": q,
            "answer": answer,
            "stopped_on_eos": stopped_on_eos,
            "generated_tokens": len(gen_ids),
        })
    print("=" * 65 + "\n", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/chat_50m_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/chat_50m_v1")
    parser.add_argument("--stage1-epochs", type=int, default=3)
    parser.add_argument("--stage2-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--stage1-lr", type=float, default=3.5e-4)
    parser.add_argument("--stage2-lr", type=float, default=1.5e-4)
    parser.add_argument("--vram-limit-gb", type=float, default=4.0)
    parser.add_argument("--skip-stage1", action="store_true")
    args = parser.parse_args()

    vram_guard = VRAMGuard(limit_gb=args.vram_limit_gb)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    tokenizer_path = ROOT / "checkpoints/bpe_stage1_vocab.json"
    tokenizer = BPETokenizer.load(tokenizer_path)

    # 50M Transformer Config
    # 16 layers, 8 heads, 512 embedding, 384 block size = ~51.65 Million parameters
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=384,
        n_layer=16,
        n_head=8,
        n_embd=512,
        dropout=0.05,
    )

    model = RobbieTransformer(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nRobbie-50M Architecture Configured:")
    print(f"  Layers: {config.n_layer} | Heads: {config.n_head} | Embedding: {config.n_embd}")
    print(f"  Context Window: {config.block_size} tokens | Vocab: {config.vocab_size} tokens")
    print(f"  Total Parameters: {param_count:,} (~{param_count / 1e6:.2f} Million parameters)\n", flush=True)

    # Load prepared data
    lang_dir = args.data_dir / "language"
    asst_dir = args.data_dir / "answers"
    lang_splits, _ = load_prepared(lang_dir, tokenizer_path)
    asst_splits, _ = load_prepared(asst_dir, tokenizer_path)

    stage1_ckpt_path = ROOT / "checkpoints/stage1_50m_base_transformer.pt"
    stage2_ckpt_path = ROOT / "checkpoints/stage2_50m_conversational_transformer.pt"

    # --- STAGE 1: Language Foundation Pretraining ---
    if not args.skip_stage1:
        print("\n>>> STAGE 1: Full-Dialogue Language Foundation Pretraining (dialogue_lm)...")
        stage1_out = args.output_dir / "stage1_language"
        train_stage(
            stage_name="Stage 1 Language Foundation",
            model=model,
            train_samples=lang_splits["train"],
            val_samples=lang_splits["validation"],
            tokenizer=tokenizer,
            epochs=args.stage1_epochs,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            base_lr=args.stage1_lr,
            min_lr=args.stage1_lr * 0.1,
            output_dir=stage1_out,
            device=device,
            vram_guard=vram_guard,
            eval_every=150,
        )

        # Load best stage 1 weights
        best_stage1 = torch.load(stage1_out / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(best_stage1["model_state_dict"])
        # Save canonical stage 1 checkpoint for Project Robbie
        torch.save(best_stage1, stage1_ckpt_path)
        print(f"Stage 1 checkpoint saved to {stage1_ckpt_path}\n", flush=True)
    else:
        if stage1_ckpt_path.is_file():
            print(f"Skipping Stage 1, loading existing weights from {stage1_ckpt_path}")
            ckpt = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

    # --- STAGE 2: Assistant Instruction SFT ---
    print("\n>>> STAGE 2: Supervised Fine-Tuning on Assistant Replies (assistant_sft)...")
    stage2_out = args.output_dir / "stage2_assistant"
    train_stage(
        stage_name="Stage 2 Assistant SFT",
        model=model,
        train_samples=asst_splits["train"],
        val_samples=asst_splits["validation"],
        tokenizer=tokenizer,
        epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        base_lr=args.stage2_lr,
        min_lr=args.stage2_lr * 0.1,
        output_dir=stage2_out,
        device=device,
        vram_guard=vram_guard,
        eval_every=150,
    )

    # Load best stage 2 weights
    best_stage2 = torch.load(stage2_out / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_stage2["model_state_dict"])
    model.to(device).eval()

    # Save canonical stage 2 checkpoint for Project Robbie
    torch.save(best_stage2, stage2_ckpt_path)
    torch.save(best_stage2, args.output_dir / "best_chat.pt")
    print(f"Stage 2 Conversational Checkpoint saved to:\n  - {stage2_ckpt_path}\n  - {args.output_dir / 'best_chat.pt'}\n", flush=True)

    # --- GENERATION EVALUATION & BENCHMARK REVIEW ---
    samples = sample_scenarios(model, tokenizer, device, max_tokens=140)
    review = assess_samples(samples)

    review_report = {
        "model_scale": "50M (~51.65 Million parameters)",
        "checkpoint": str(stage2_ckpt_path),
        "val_loss": best_stage2["val_loss"],
        "val_perplexity": best_stage2["val_perplexity"],
        "vram_limit_gb": args.vram_limit_gb,
        "vram_slowdowns_triggered": vram_guard.slowdowns,
        "generation_review": review,
    }
    (args.output_dir / "chat_review.json").write_text(json.dumps(review_report, indent=2), encoding="utf-8")
    print(f"Quality review written to {args.output_dir / 'chat_review.json'}")
    print(f"Status: {review['status']} (Failed samples: {review['failed_samples']}/{review['sample_count']})")
    print("\n50M Model Training and Alignment Complete!")


if __name__ == "__main__":
    main()
