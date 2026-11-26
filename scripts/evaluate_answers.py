"""Print actual model answers to unseen wording, bypassing all built-in replies."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from robbie.core.models.transformer import RobbieTransformer
from robbie.core.tokenizer.bpe_tokenizer import BPETokenizer
from robbie.core.conversation.engine import ChatEngine


QUESTIONS = [
    "Hello, how can you help me today?",
    "What is RAM used for?",
    "Do RAM and an SSD do the same job?",
    "What distinguishes video memory from system memory?",
    "What is the purpose of a branch in Git?",
    "What happens when I make a commit?",
    "What could make my computer run slowly?",
    "Is a single high memory reading enough to diagnose a leak?",
    "Can a computer still have a GPU if CUDA is not available?",
    "What should I check before closing a process?",
    "Why would I use a separate Python environment for each project?",
    "Does a foreground window title tell you every browser tab?",
    "What is my RAM usage right now?",
    "How much memory is available?\nComputer observation: RAM free=7 GB; total=24 GB.",
    "How much memory is available?\nComputer observation: RAM free=13 GB; total=32 GB.",
    "Thanks, that helped.",
]


def evaluate(checkpoint_path, output_path=None, max_tokens=100, device=None):
    torch.set_num_threads(4)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BPETokenizer.load(ROOT / "checkpoints/bpe_stage1_vocab.json")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = RobbieTransformer(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    rows = []
    print(f"Checkpoint: {checkpoint_path}\nDevice: {device}", flush=True)
    for question in QUESTIONS:
        # Each question starts with fresh context. These are model outputs, not
        # the deterministic computer-report or personality branches in respond().
        engine = ChatEngine(model, tokenizer, device=device, max_tokens=max_tokens,
                            temperature=0, repetition_penalty=1.0)
        start = time.perf_counter()
        answer = engine.generate_reply(question)
        elapsed = time.perf_counter() - start
        rows.append({"question": question, "answer": answer, "seconds": round(elapsed, 3)})
        print(f"\nQuestion: {question}\nRobbie: {answer}\n({elapsed:.2f}s)", flush=True)
    report = {"checkpoint": str(checkpoint_path), "mode": "model only; greedy decoding",
              "note": "Manual answer review, not an accuracy benchmark", "answers": rows}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport saved: {output_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output, args.max_tokens, args.device)


if __name__ == "__main__":
    main()
