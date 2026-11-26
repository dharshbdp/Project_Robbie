"""Train the basic-conversation curriculum, preserving and resuming both stages."""

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch


def stage_status(directory, epochs):
    path = directory / "last.pt"
    if not path.is_file():
        return "new"
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["training_config"]["epochs"] != epochs:
        raise ValueError("The requested epochs differ from the existing run. Restore the original setting or choose a new --output-dir.")
    return "complete" if state["epoch"] >= epochs else "resume"


def command_for(stage, args, status):
    output = args.output_dir / stage
    data = args.data_dir / ("dialogue" if stage == "language" else "answers")
    source = args.checkpoint if stage == "language" else args.output_dir / "language/best_chat.pt"
    command = [sys.executable, "-u", str(ROOT / "scripts/train_chat.py"),
               "--data-dir", str(data), "--profile", "basic"]
    if args.dry_run:
        source = source if source.is_file() else args.checkpoint
        return command + ["--checkpoint", str(source), "--batch-size", str(args.batch_size), "--dry-run"]
    if status == "resume":
        return command + ["--resume", str(output / "last.pt")]
    return command + ["--checkpoint", str(source), "--output-dir", str(output),
        "--epochs", str(args.language_epochs if stage == "language" else args.reply_epochs),
        "--learning-rate", str(2e-4 if stage == "language" else 1e-4),
        "--batch-size", str(args.batch_size), "--accumulation-steps", str(args.accumulation_steps),
        "--log-every", "5", "--eval-every", "100", "--save-every", "50"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/clean_chat_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/clean_chat_v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/stage1_25m_base_transformer.pt")
    parser.add_argument("--language-epochs", type=int, default=8)
    parser.add_argument("--reply-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--stage", choices=["all", "language", "replies"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.language_epochs, args.reply_epochs, args.batch_size, args.accumulation_steps) < 1:
        parser.error("Epochs and batch settings must be positive")
    if not args.checkpoint.is_file():
        parser.error("The initial base checkpoint does not exist")
    for name in ("dialogue", "answers"):
        path = args.data_dir / name / "manifest.json"
        if not path.is_file():
            parser.error("Prepare the curriculum first with scripts/prepare_clean_chat.py")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = "dialogue_lm" if name == "dialogue" else "assistant_sft"
        if manifest.get("objective") != expected or manifest.get("dialogue_format") != "assistant-eos-v1":
            parser.error(f"Unexpected curriculum format or objective in {path}")
    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
        print("\nWaiting for the current training stage to save. The next stage will not start.", flush=True)
    signal.signal(signal.SIGINT, request_stop)
    stages = ["language", "replies"] if args.stage == "all" else [args.stage]
    for stage in stages:
        epochs = args.language_epochs if stage == "language" else args.reply_epochs
        directory = args.output_dir / stage
        status = "new" if args.dry_run else stage_status(directory, epochs)
        if status == "complete":
            if not (directory / "best_chat.pt").is_file():
                raise ValueError(f"Completed stage has no best checkpoint: {directory}")
            print(f"Stage {stage} already complete; preserving its checkpoint.", flush=True)
            continue
        if stage == "replies" and status == "new" and not args.dry_run:
            if stage_status(args.output_dir / "language", args.language_epochs) != "complete":
                raise ValueError("Finish the language stage before starting reply tuning")
        print(f"\n=== {'CHECK' if args.dry_run else 'TRAIN'}: {stage.upper()} ===", flush=True)
        child = subprocess.Popen(command_for(stage, args, status), cwd=ROOT)
        code = child.wait()
        if code:
            print(f"Stage {stage} stopped with exit code {code}; no later stage was started.", flush=True)
            return code
        if stopped or (not args.dry_run and stage_status(directory, epochs) != "complete"):
            print("Paused. Run the same command again to resume the saved stage.", flush=True)
            return 0
    if args.dry_run:
        print("Both data objectives passed setup checks. No optimizer updates or model checkpoints saved.", flush=True)
    else:
        print(f"\nRequested stages finished. Candidate checkpoint: {args.output_dir / stages[-1] / 'best_chat.pt'}\n"
              "Review basic-chat samples for relevance, follow-ups, repetition, and EOS stopping before accepting it.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
