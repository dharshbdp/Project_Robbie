"""Prepare comprehensive 300M curriculum for Robbie.

Combines:
1. alespalla/chatbot_instruction_prompts (258k instruction-response pairs)
2. Robbie core identity, persona, and etiquette scenarios
3. Computer architecture, hardware, and programming Q&A
4. Clean conversational dialogue from Dolly-15k and No-Robots
Outputs processed assistant-only SFT (Stage 2) and language foundation (Stage 1).
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from robbie.core.data.chat_data import (
    FORMAT_VERSION,
    digest_file,
    encode_conversation,
)
from robbie.core.tokenizer.bpe_tokenizer import BPETokenizer


def normalize_text(text: str) -> str:
    """Normalize unicode quotes, dashes, and artifacts for BPETokenizer."""
    if not text:
        return ""
    for c in ["\u2018", "\u2019", "\u0060", "\u00b4", "\ufffd"]:
        text = text.replace(c, "'")
    for c in ["\u201c", "\u201d", "\u00ab", "\u00bb"]:
        text = text.replace(c, '"')
    for c in ["\u2013", "\u2014"]:
        text = text.replace(c, "-")
    text = text.replace("\u2026", "...").replace("\u00a0", " ")
    # Clean up excess whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_curated_conversations(tokenizer, context: int):
    """Load core identity, computer Q&A, and Dolly/No-Robots datasets."""
    conversations = []

    # 1. Robbie Curated Persona & Identity
    curated_path = ROOT / "data/curated/robbie_50m_curated.json"
    if curated_path.is_file():
        data = json.loads(curated_path.read_text(encoding="utf-8"))
        for item in data:
            conversations.append({"messages": item["conversation"], "source": "robbie_curated", "weight": 30})

    # 2. Chat Basics
    basics_path = ROOT / "data/curated/chat_basics.json"
    if basics_path.is_file():
        data = json.loads(basics_path.read_text(encoding="utf-8"))
        for item in data:
            conversations.append({"messages": item["conversation"], "source": "chat_basics", "weight": 20})

    # 3. Computer Systems & Programming Answers
    comp_path = ROOT / "data/curated/computer_answers.json"
    if comp_path.is_file():
        data = json.loads(comp_path.read_text(encoding="utf-8"))
        for item in data:
            ans = item["answer"]
            for q in item["questions"]:
                conv = [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
                conversations.append({"messages": conv, "source": "computer_answers", "weight": 25})

    # 4. Dolly & No-Robots Curated
    dolly_norobots_path = ROOT / "data/curated/dolly_and_norobots_curated.json"
    if dolly_norobots_path.is_file():
        data = json.loads(dolly_norobots_path.read_text(encoding="utf-8"))
        for item in data[:6000]:
            conversations.append({
                "messages": item["conversation"],
                "source": item.get("source", "hf_curated"),
                "weight": 2,
            })

    # 5. Assistant Dialogues
    asst_path = ROOT / "data/raw/stage2_dialogue/assistant_dialogues.jsonl"
    if asst_path.is_file():
        for line in asst_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                conversations.append({"messages": item["conversation"], "source": "assistant_dialogues", "weight": 3})

    return conversations


def load_alespalla_instructions(tokenizer, context: int, max_examples: int = 35000):
    """Load, clean, and filter instruction prompts from alespalla/chatbot_instruction_prompts."""
    from datasets import load_dataset
    print(f"Loading 'alespalla/chatbot_instruction_prompts' from Hugging Face cache...", flush=True)
    ds = load_dataset("alespalla/chatbot_instruction_prompts", split="train")
    print(f"Total raw items available: {len(ds):,}", flush=True)

    examples = []
    # Deterministic stride/shuffle for diversity
    indices = list(range(len(ds)))
    random.seed(42)
    random.shuffle(indices)

    skipped_len = 0
    skipped_empty = 0

    for idx in indices:
        if len(examples) >= max_examples:
            break
        row = ds[idx]
        p = normalize_text(row.get("prompt", ""))
        r = normalize_text(row.get("response", ""))

        if not p or not r or len(p) < 3 or len(r) < 3:
            skipped_empty += 1
            continue

        # Basic length heuristic before tokenization
        if len(p) + len(r) > 2500:
            skipped_len += 1
            continue

        msgs = [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
        encoded = encode_conversation(msgs, tokenizer, context, assistant_eos=True)
        if encoded:
            for ex in encoded:
                ex["source"] = "alespalla_instructions"
                examples.append(ex)
                if len(examples) >= max_examples:
                    break
        else:
            skipped_len += 1

        if len(examples) % 5000 == 0 and len(examples) > 0:
            print(f"  Processed {len(examples):,} valid alespalla examples...", flush=True)

    print(f"Loaded {len(examples):,} cleaned examples from alespalla (skipped: {skipped_len:,} length, {skipped_empty:,} empty)", flush=True)
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/chat_300m_v1")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--max-alespalla", type=int, default=35000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer_path = ROOT / "checkpoints/bpe_stage1_vocab.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer vocabulary not found at {tokenizer_path}")
    tokenizer = BPETokenizer.load(tokenizer_path)

    print("\n1. Loading curated core persona & domain dialogues...", flush=True)
    raw_convs = load_curated_conversations(tokenizer, args.context)
    print(f"Loaded {len(raw_convs):,} base curated items", flush=True)

    curated_examples = []
    for item in raw_convs:
        weight = item.get("weight", 1)
        encoded = encode_conversation(item["messages"], tokenizer, args.context, assistant_eos=True)
        if encoded:
            for ex in encoded:
                ex["source"] = item["source"]
                for _ in range(weight):
                    curated_examples.append(dict(ex))
    print(f"Produced {len(curated_examples):,} weighted curated examples", flush=True)

    print("\n2. Loading and filtering alespalla/chatbot_instruction_prompts...", flush=True)
    alespalla_examples = load_alespalla_instructions(tokenizer, args.context, max_examples=args.max_alespalla)

    # 3. Clean everyday dialogues from clean_chat_v1
    clean_chat_path = ROOT / "data/processed/clean_chat_v1/answers/train.pt"
    clean_examples = []
    if clean_chat_path.is_file():
        clean_data = torch.load(clean_chat_path, map_location="cpu", weights_only=False)
        for ex in clean_data:
            if len(ex["tokens"]) <= args.context + 1:
                clean_examples.append({"tokens": ex["tokens"], "answer_start": ex["answer_start"], "source": "clean_smoltalk"})
        print(f"Loaded {len(clean_examples):,} clean everyday dialogue examples from clean_chat_v1", flush=True)

    all_train = curated_examples + alespalla_examples + clean_examples
    random.shuffle(all_train)

    n_val = min(1500, max(300, int(len(all_train) * 0.04)))
    val_samples = all_train[:n_val]
    train_samples = all_train[n_val:]

    print(f"\nFinal 300M Curriculum:")
    print(f"  Training set: {len(train_samples):,} examples")
    print(f"  Validation set: {len(val_samples):,} examples")
    print(f"  Context window: {args.context} tokens\n", flush=True)

    # Prepare Stage 1 and Stage 2 datasets
    for stage, obj in [("language", "dialogue_lm"), ("answers", "assistant_sft")]:
        stage_dir = args.output_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        if obj == "dialogue_lm":
            stage_train = [{**s, "answer_start": 0} for s in train_samples]
            stage_val = [{**s, "answer_start": 0} for s in val_samples]
        else:
            stage_train = train_samples
            stage_val = val_samples

        train_path = stage_dir / "train.pt"
        val_path = stage_dir / "validation.pt"
        test_path = stage_dir / "test.pt"

        torch.save(stage_train, train_path)
        torch.save(stage_val, val_path)
        torch.save(stage_val[:300], test_path)

        manifest = {
            "format_version": FORMAT_VERSION,
            "dialogue_format": "assistant-eos-v1",
            "context": args.context,
            "objective": obj,
            "tokenizer_sha256": digest_file(tokenizer_path),
            "splits": {
                "train": {
                    "examples": len(stage_train),
                    "sha256": digest_file(train_path),
                    "sources": dict(Counter(s.get("source", "unknown") for s in stage_train)),
                },
                "validation": {
                    "examples": len(stage_val),
                    "sha256": digest_file(val_path),
                    "sources": dict(Counter(s.get("source", "unknown") for s in stage_val)),
                },
                "test": {
                    "examples": len(stage_val[:300]),
                    "sha256": digest_file(test_path),
                },
            },
        }
        (stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {stage} split to {stage_dir} (Train: {len(stage_train):,}, Val: {len(stage_val):,})", flush=True)

    print("\nDataset preparation for Robbie-300M completed successfully!", flush=True)


if __name__ == "__main__":
    main()
