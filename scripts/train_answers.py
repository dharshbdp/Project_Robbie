"""Fine-tune an existing Robbie checkpoint on concise computer answers.

This is a small supervised pilot, not broad pretraining. Original checkpoints
are preserved, and validation uses question wording absent from training.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from robbie.core.data.answer_dataset import AnswerDataset
from robbie.core.models.transformer import RobbieTransformer
from robbie.core.tokenizer.bpe_tokenizer import BPETokenizer
from robbie.core.training.trainer import Trainer, TrainingConfig


def pair(question, answer):
    return {"conversation": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]}


def build_data(corpus_path):
    specifications = json.loads(corpus_path.read_text(encoding="utf-8"))
    train, val = [], []
    for item in specifications:
        train.extend(pair(q, item["answer"]) for q in item["questions"][:-1])
        val.append(pair(item["questions"][-1], item["answer"]))
    # Learn to read observations rather than substitute memorized machine values.
    # Validation uses different question wording and disjoint state combinations.
    seen_states = set()
    for split, seed, count in ((train, 42, 144), (val, 1042, 24)):
        rng = random.Random(seed)
        for _ in range(count):
            while True:
                total = rng.choice([8, 16, 24, 32, 48, 64])
                free = rng.randint(1, total - 1)
                if (free, total) not in seen_states:
                    seen_states.add((free, total))
                    break
            question = ("How much RAM is free?" if split is train else "Tell me the available RAM.")
            observation = f"Computer observation: RAM free={free} GB; total={total} GB."
            split.append(pair(question + "\n" + observation,
                              f"You have {free} GB of RAM free out of {total} GB total."))
    # Follow-up explanations train against the same dialogue format used in chat.
    for item in specifications[:40]:
        train.append({"conversation": [
            {"role": "user", "content": item["questions"][0]},
            {"role": "assistant", "content": item["answer"]},
            {"role": "user", "content": "Can you summarize that in one sentence?"},
            {"role": "assistant", "content": item["answer"].split(". ")[0].rstrip(".") + "."},
        ]})
    return train, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/stage2_25m_conversational_transformer.pt")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/curated/computer_answers.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/answers_25m")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=5, help="Print progress every N batches")
    parser.add_argument("--dry-run", action="store_true", help="Check data and one forward pass without training or saving weights")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.log_every < 1 or args.context < 2:
        parser.error("epochs, batch size, log interval, and context must be positive")
    if args.output_dir.resolve() == args.checkpoint.parent.resolve():
        parser.error("Use a separate output directory to preserve original checkpoints")
    if (args.output_dir / "best_answers.pt").exists() and not args.dry_run:
        parser.error("Output already contains a checkpoint; choose a new output directory")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    tokenizer = BPETokenizer.load(ROOT / "checkpoints/bpe_stage1_vocab.json")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    old_config = checkpoint["config"]
    config = replace(old_config, block_size=args.context, dropout=0.05)
    model = RobbieTransformer(config)
    state = checkpoint["model_state_dict"]
    old_positions = state["wpe.weight"]
    with torch.no_grad():
        count = min(args.context, old_positions.shape[0])
        model.wpe.weight[:count].copy_(old_positions[:count])
    # Causal masks are deterministic buffers; recreate them at the new length.
    compatible = {k: v for k, v in state.items() if k != "wpe.weight" and not k.endswith(".attn.bias")}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    allowed_missing = {"wpe.weight"} | {f"blocks.{i}.attn.bias" for i in range(config.n_layer)}
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(f"Incompatible checkpoint: {missing}, {unexpected}")
    train, val = build_data(args.corpus)
    train_ds = AnswerDataset(train, tokenizer, args.context)
    val_ds = AnswerDataset(val, tokenizer, args.context)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Examples: train={len(train_ds)}, validation={len(val_ds)}; "
          f"skipped={train_ds.skipped}/{val_ds.skipped}", flush=True)
    if args.dry_run:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        x, y = train_ds[0]
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
            _, loss = model(x[None].to(device), y[None].to(device))
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss in setup check")
        print(f"Setup check passed on {device}; no training performed.", flush=True)
        return
    trainer = Trainer(model, train_ds, val_ds, TrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        checkpoint_dir=str(args.output_dir), best_model_name="best_answers.pt",
        use_amp=True, early_stopping_patience=5,
        log_interval=args.log_every,
    ))
    history = trainer.fit()
    metadata = {
        "purpose": "Small computer-answer fine-tuning pilot; not general assistant training",
        "source_checkpoint": str(args.checkpoint), "corpus": str(args.corpus),
        "train_examples": len(train_ds), "validation_examples": len(val_ds),
        "train_skipped": train_ds.skipped, "validation_skipped": val_ds.skipped,
        "history": history,
    }
    (args.output_dir / "answer_training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {args.output_dir / 'best_answers.pt'}", flush=True)


if __name__ == "__main__":
    main()
