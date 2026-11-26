"""Prepare unified, high-quality conversational training curriculum for Robbie-50M.

Combines:
1. Curated Robbie identity, persona, and everyday dialogue scenarios
2. Comprehensive computer systems, hardware, and programming Q&A
3. Alpaca Clean & Dolly dialogues
4. SmolTalk & OpenAssistant rich instruction dataset
Outputs both full-dialogue language modeling (Stage 1) and assistant-only SFT (Stage 2).
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from robbie.core.data.chat_data import (
    FORMAT_VERSION, digest_file, encode_conversation, prompt_group, split_for_group
)
from robbie.core.tokenizer.bpe_tokenizer import BPETokenizer


def load_curated_conversations(tokenizer, context):
    conversations = []

    # 1. Robbie 50m Curated
    curated_path = ROOT / "data/curated/robbie_50m_curated.json"
    if curated_path.is_file():
        data = json.loads(curated_path.read_text(encoding="utf-8"))
        for item in data:
            conversations.append({"messages": item["conversation"], "source": "robbie_curated", "weight": 35})

    # 2. Chat basics
    basics_path = ROOT / "data/curated/chat_basics.json"
    if basics_path.is_file():
        data = json.loads(basics_path.read_text(encoding="utf-8"))
        for item in data:
            conversations.append({"messages": item["conversation"], "source": "chat_basics", "weight": 20})

    # 3. Computer answers
    comp_path = ROOT / "data/curated/computer_answers.json"
    if comp_path.is_file():
        data = json.loads(comp_path.read_text(encoding="utf-8"))
        for item in data:
            ans = item["answer"]
            for q in item["questions"]:
                conv = [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
                conversations.append({"messages": conv, "source": "computer_answers", "weight": 30})

    # 4. Assistant dialogues
    asst_path = ROOT / "data/raw/stage2_dialogue/assistant_dialogues.jsonl"
    if asst_path.is_file():
        for line in asst_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                conversations.append({"messages": item["conversation"], "source": "assistant_dialogues", "weight": 4})

    # 5. Alpaca Clean (sample top clean dialogues)
    alpaca_path = ROOT / "data/raw/stage2_dialogue/alpaca_clean_dialogues.jsonl"
    if alpaca_path.is_file():
        lines = alpaca_path.read_text(encoding="utf-8").splitlines()
        for line in lines[:1000]:
            if line.strip():
                item = json.loads(line)
                conversations.append({"messages": item["conversation"], "source": "alpaca_clean", "weight": 1})

    # 6. Full HuggingFaceH4/no_robots & Databricks Dolly 15k Curated
    dolly_norobots_path = ROOT / "data/curated/dolly_and_norobots_curated.json"
    if dolly_norobots_path.is_file():
        data = json.loads(dolly_norobots_path.read_text(encoding="utf-8"))
        for item in data[:6000]:
            conversations.append({
                "messages": item["conversation"],
                "source": item.get("source", "hf_curated"),
                "weight": 2,
            })
    else:
        dolly_path = ROOT / "data/raw/stage2_dialogue/dolly_dialogues.jsonl"
        if dolly_path.is_file():
            lines = dolly_path.read_text(encoding="utf-8").splitlines()
            for line in lines[:800]:
                if line.strip():
                    item = json.loads(line)
                    conversations.append({"messages": item["conversation"], "source": "dolly", "weight": 1})

    return conversations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/chat_50m_v1")
    parser.add_argument("--context", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-smoltalk", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer_path = ROOT / "checkpoints/bpe_stage1_vocab.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer vocabulary not found at {tokenizer_path}")
    tokenizer = BPETokenizer.load(tokenizer_path)

    print("Building curated and instruction conversations...", flush=True)
    raw_convs = load_curated_conversations(tokenizer, args.context)
    print(f"Loaded {len(raw_convs):,} base curated items", flush=True)

    # Encode with weighting
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

    # Load high-quality clean everyday conversations from clean_chat_v1
    clean_chat_path = ROOT / "data/processed/clean_chat_v1/answers/train.pt"
    clean_examples = []
    if clean_chat_path.is_file():
        clean_data = torch.load(clean_chat_path, map_location="cpu", weights_only=False)
        for ex in clean_data:
            if len(ex["tokens"]) <= args.context + 1:
                clean_examples.append({"tokens": ex["tokens"], "answer_start": ex["answer_start"], "source": "clean_smoltalk"})
        print(f"Loaded {len(clean_examples):,} clean everyday dialogue examples from clean_chat_v1", flush=True)

    # Load and filter existing chat_v1 (filtering out synthetic prompt benchmark artifacts)
    existing_chat_v1 = ROOT / "data/processed/chat_v1/train.pt"
    chat_v1_examples = []
    NOISY_PATTERNS = [
        "keywords [", "keyword [", "include keywords", "contain keywords",
        "at least 3 sentences", "at least 4 sentences", "word count", "character count",
        "park ranger", "wise wizard", "elder approaches", "riddle me",
        "five year old", "characters spell", "exact phrase", "must contain the words",
        "dog, tree, park", "dog, walk, park", "stay in character as a",
    ]
    if args.include_smoltalk and existing_chat_v1.is_file():
        v1_data = torch.load(existing_chat_v1, map_location="cpu", weights_only=False)
        print(f"Inspecting {len(v1_data):,} examples from chat_v1 for noisy benchmark artifacts...", flush=True)
        for ex in v1_data:
            if len(ex["tokens"]) <= args.context + 1:
                decoded = tokenizer.decode(ex["tokens"].tolist()).lower()
                if not any(pat in decoded for pat in NOISY_PATTERNS):
                    chat_v1_examples.append({"tokens": ex["tokens"], "answer_start": ex["answer_start"], "source": "smoltalk_filtered"})
                    if len(chat_v1_examples) >= 4000:
                        break
        print(f"Kept {len(chat_v1_examples):,} high-quality filtered chat_v1 examples within context {args.context}", flush=True)

    # Combine all examples
    all_train = curated_examples + clean_examples + chat_v1_examples
    random.shuffle(all_train)

    # Split into train (95%) and validation (5%)
    n_val = min(1000, max(200, int(len(all_train) * 0.05)))
    val_samples = all_train[:n_val]
    train_samples = all_train[n_val:]

    print(f"Final training set: {len(train_samples):,} examples | validation: {len(val_samples):,} examples", flush=True)

    # Prepare Stage 1 (language: answer_start=0) and Stage 2 (answers: answer_start=actual)
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
        torch.save(stage_val[:200], test_path)

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
                    "examples": len(stage_val[:200]),
                    "sha256": digest_file(test_path),
                },
            },
        }
        (stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {stage} stage to {stage_dir} (Train: {len(stage_train):,}, Val: {len(stage_val):,})", flush=True)

    print("\nDataset preparation complete! Ready for 50M model training.", flush=True)


if __name__ == "__main__":
    main()
