"""Fine-tune Robbie on prepared conversations. Training starts only when invoked."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from robbie.core.data.chat_data import collate_answers, digest_file, load_prepared
from robbie.core.models.transformer import RobbieTransformer
from robbie.core.tokenizer.bpe_tokenizer import BPETokenizer
from robbie.core.training.chat_training import ChatTrainingConfig, ChatTrainingRun, load_initial_model
from robbie.core.training.generation_quality import assess_samples
from robbie.core.training.chat_profiles import basic_samples


def sample_answers(model, tokenizer, prompts, max_tokens=96):
    """Raw greedy generation with the same role formatting as training."""
    device = next(model.parameters()).device
    was_training = model.training
    answers = []
    try:
        for prompt in prompts:
            ids = tokenizer.encode(f"User: {prompt}\nRobbie:", add_special_tokens=False)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            output = model.generate(x, max_new_tokens=max_tokens, temperature=0,
                                    stop_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(output[0, len(ids):].tolist(), skip_special_tokens=True).strip()
            answers.append({"question": prompt, "answer": answer})
        return answers
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/chat_v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/stage1_25m_base_transformer.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/chat_25m_v1")
    parser.add_argument("--resume", type=Path, help="Resume last.pt using its original settings and run directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--profile", choices=["general", "basic"], help="Generation review scenarios (restored automatically on resume)")
    parser.add_argument("--dry-run", action="store_true", help="Check a full-length forward/backward pass without updating or saving weights")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.eval_every, args.save_every, args.log_every) < 1 or args.learning_rate <= 0:
        parser.error("Training counts and learning rate must be positive")
    torch.set_num_threads(4)
    tokenizer_path = ROOT / "checkpoints/bpe_stage1_vocab.json"
    tokenizer = BPETokenizer.load(tokenizer_path)
    splits, manifest = load_prepared(args.data_dir, tokenizer_path)
    fingerprint = digest_file(args.data_dir / "manifest.json")
    config = ChatTrainingConfig(epochs=args.epochs, batch_size=args.batch_size,
                               accumulation_steps=args.accumulation_steps, learning_rate=args.learning_rate,
                               eval_every=args.eval_every, save_every=args.save_every,
                               log_every=args.log_every, device=args.device)
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = ChatTrainingConfig(**payload["training_config"])
        args.output_dir = args.resume.resolve().parent
        saved_run = json.loads((args.output_dir / "run.json").read_text(encoding="utf-8"))
        saved_profile = saved_run.get("profile", "general")
        if args.profile and args.profile != saved_profile:
            parser.error("Resume profile differs from the original run")
        args.profile = saved_profile
        model = RobbieTransformer(payload["config"])
        model.load_state_dict(payload["model_state_dict"])
        if fingerprint != payload["data_fingerprint"]:
            parser.error("Prepared data differs from resumed run. Supply its original --data-dir.")
    else:
        args.profile = args.profile or "general"
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        model = load_initial_model(args.checkpoint, manifest["context"])
        if (args.output_dir / "last.pt").exists() and not args.dry_run:
            parser.error("A run already exists. Use --resume or choose a different --output-dir.")
    model.dialogue_format = manifest.get("dialogue_format", "legacy")
    model.training_objective = manifest.get("objective", "assistant_sft")
    if config.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Fix the CUDA environment or explicitly use --device cpu for a new run.")
    if model.config.vocab_size != tokenizer.vocab_size or model.config.block_size != manifest["context"]:
        parser.error("Model, tokenizer, and prepared context are incompatible")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters | context {model.config.block_size}\n"
          + " | ".join(f"{name}: {len(data):,}" for name, data in splits.items()), flush=True)
    if args.dry_run:
        model.to(config.device).train()
        samples = sorted(splits["train"], key=lambda item: len(item["tokens"]), reverse=True)[:config.batch_size]
        x, y = collate_answers(samples, tokenizer.pad_token_id)
        amp = config.device == "cuda" and config.amp
        dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(config.device, dtype=dtype, enabled=amp):
            _, loss = model(x.to(config.device), y.to(config.device))
        if not torch.isfinite(loss):
            raise ValueError("Non-finite setup loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Non-finite setup gradients")
        memory = f" Peak allocated VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB." if amp else ""
        print(f"Setup passed on {config.device}; batch shape {tuple(x.shape)}, loss {loss.item():.4f}. "
              f"Forward/backward only; no optimizer updates or saved weights.{memory}", flush=True)
        return
    def samples(current_model, step, directory):
        rows = basic_samples(current_model, tokenizer) if args.profile == "basic" else sample_answers(current_model, tokenizer, [
            "Hello! What can you help me with?", "Explain why computers need RAM in two sentences.",
            "I am learning Python. Suggest a small beginner project.",
        ])
        (directory / f"samples_{step:06d}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        review = assess_samples(rows)
        (directory / f"generation_{step:06d}.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
        for row in rows:
            print(f"SAMPLE | {row['question']}\nRobbie: {row['answer']}", flush=True)
        return review
    run = ChatTrainingRun(model, splits, tokenizer, config, args.output_dir, fingerprint, samples)
    if args.resume:
        run.resume(args.resume)
        print(f"Resumed update {run.step}, epoch {run.epoch + 1}, example cursor {run.cursor}", flush=True)
    else:
        (args.output_dir / "run.json").write_text(json.dumps({
            "source_checkpoint": str(args.checkpoint.resolve()), "source_sha256": digest_file(args.checkpoint), "profile": args.profile,
            "data_dir": str(args.data_dir.resolve()), "data_fingerprint": fingerprint,
            "training_config": asdict(config), "data_manifest": manifest,
        }, indent=2), encoding="utf-8")
    def request_stop(signum, frame):
        run.stop_requested = True
        print("\nStop requested. Finishing the current update and saving; please wait.", flush=True)
    signal.signal(signal.SIGINT, request_stop)
    run.fit()


if __name__ == "__main__":
    main()
