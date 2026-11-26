"""Evaluate unseen conversation loss and actual multi-turn model replies."""

import argparse
import json
from pathlib import Path

from chat_model import ROOT, load_chat

import torch

from robbie.core.data.chat_data import collate_answers, digest_file, load_prepared
from robbie.core.training.generation_quality import assess_samples
from robbie.core.training.chat_profiles import BASIC_SCENARIOS


SCENARIOS = [
    ["Hi! What can you help me with?", "I want to learn Python. Where should I begin?"],
    ["What is the difference between RAM and storage?", "Explain that more simply."],
    ["My app freezes when I open a large file. What should I check?", "Should I close other applications first?"],
    ["I have a folder of photos to organize. Suggest three steps.", "Can you summarize those steps in one sentence?"],
    ["What is using the most RAM on my computer right now?"],
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/chat_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=["general", "basic"], default="general")
    args = parser.parse_args()
    chat = load_chat(args.checkpoint, temperature=0)
    splits, manifest = load_prepared(args.data_dir, ROOT / "checkpoints/bpe_stage1_vocab.json")
    if manifest["context"] != chat.model.config.block_size:
        parser.error("Evaluation data context differs from the checkpoint")
    trained = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if trained.get("data_fingerprint") != digest_file(args.data_dir / "manifest.json"):
        parser.error("Evaluation data does not match the held-out split of this training run")
    weighted_loss, tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(splits["test"]), 4):
            x, y = collate_answers(splits["test"][start:start + 4], chat.tokenizer.pad_token_id)
            count = int((y != -100).sum())
            _, loss = chat.model(x.to(chat.device), y.to(chat.device))
            weighted_loss += loss.item() * count
            tokens += count
    report = {"checkpoint": str(args.checkpoint), "test_answer_loss": weighted_loss / tokens,
              "test_examples": len(splits["test"]), "test_answer_tokens": tokens,
              "note": "Loss measures next-token prediction, not answer accuracy. Review generated replies for correctness and instruction following.",
              "scenarios": []}
    print(f"Held-out test answer loss: {report['test_answer_loss']:.4f}", flush=True)
    for scenario in (BASIC_SCENARIOS if args.profile == "basic" else SCENARIOS):
        chat.messages.clear()
        turns = []
        for question in scenario:
            answer = chat.reply(question)
            turns.append({"question": question, "answer": answer, **chat.last_generation})
            print(f"\nYou: {question}\nRobbie: {answer}", flush=True)
        report["scenarios"].append(turns)
    report["generation_review"] = assess_samples([turn for turns in report["scenarios"] for turn in turns])
    print("\nGeneration check:", report["generation_review"]["status"], flush=True)
    output = args.output or args.checkpoint.parent / "chat_review.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {output}", flush=True)


if __name__ == "__main__":
    main()
