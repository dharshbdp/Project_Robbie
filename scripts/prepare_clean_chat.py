"""Build a small, short-reply conversation curriculum from cached public data."""

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

import pyarrow.parquet as pq
import torch

from robbie.core.data.chat_data import FORMAT_VERSION, digest_file, encode_conversation, prompt_group
from robbie.core.tokenizer.bpe_tokenizer import BPETokenizer


REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
REPO = "HuggingFaceTB/smol-smoltalk"
SOURCE = "everyday-conversations"
GREETINGS = {"hi", "hello", "hey", "hey there", "hi there", "hello there"}
PUNCTUATION = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                             "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "})


def has_starter(messages):
    return len(messages) >= 4 and " ".join(re.findall(r"\w+", messages[0]["content"].casefold())) in GREETINGS


def conversation_group(messages):
    # The source commonly starts unrelated conversations with the same greeting.
    # Group on the first topic rather than placing every "Hi" in one split.
    return prompt_group(messages[2:] if has_starter(messages) else messages)


def normalize_and_filter(messages, tokenizer):
    normalized = []
    if len(messages) < 2 or len(messages) % 2:
        return None
    for index, message in enumerate(messages):
        role = "user" if index % 2 == 0 else "assistant"
        if message["role"] != role:
            return None  # No dropping system directives or intervening turns.
        text = message["content"].translate(PUNCTUATION).strip()
        if not text or len(text.split()) > 40 or "```" in text:
            return None
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > (128 if role == "assistant" else 160):
            return None
        if ids.count(tokenizer.unk_token_id) / len(ids) > 0.01:
            return None
        normalized.append({"role": role, "content": text})
    return normalized


def write_stage(directory, records, groups, tokenizer_path, provenance, context, objective,
                split_method=None, limits=None):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": FORMAT_VERSION, "dialogue_format": "assistant-eos-v1",
                "context": context, "objective": objective,
                "tokenizer_sha256": digest_file(tokenizer_path), "provenance": provenance,
                "split_method": split_method or "First substantive user prompt grouped by SHA256; 90/5/5. Shared greeting targets are training-only.",
                "limits": limits or "Narrow basic-conversation curriculum, not a factual accuracy benchmark or broad pretraining corpus.",
                "splits": {}}
    for name, examples in records.items():
        samples = [{**example, "answer_start": 0} for example in examples] if objective == "dialogue_lm" else examples
        path = directory / (name + ".pt")
        torch.save(samples, path)
        manifest["splits"][name] = {"examples": len(samples), "sha256": digest_file(path),
            "groups": len(groups[name]), "supervised_tokens": sum(len(s["tokens"]) - 1 - s["answer_start"] for s in samples),
            "input_tokens": sum(len(s["tokens"]) - 1 for s in samples),
            "sources": dict(Counter(s["source"] for s in samples))}
        (directory / (name + "_groups.json")).write_text(json.dumps(sorted(groups[name])), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(objective, json.dumps(manifest["splits"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/clean_chat_v1")
    parser.add_argument("--context", type=int, default=512)
    args = parser.parse_args()
    if args.context < 256:
        parser.error("Use a context of at least 256 tokens to retain follow-up context")
    if any((args.output_dir / stage / "manifest.json").exists() for stage in ("dialogue", "answers")):
        parser.error("Prepared curriculum exists. Choose a different --output-dir to preserve active runs.")
    raw_dir = ROOT / "data/raw/chat/HuggingFaceTB_smol-smoltalk" / REVISION
    paths = [raw_dir / f"train-{i:05d}-of-00004.parquet" for i in range(4)]
    if not all(path.is_file() for path in paths):
        parser.error("Pinned source files are missing. Run scripts/prepare_chat.py to download them first.")
    tokenizer_path = ROOT / "checkpoints/bpe_stage1_vocab.json"
    tokenizer = BPETokenizer.load(tokenizer_path)
    seed_path = ROOT / "data/curated/chat_basics.json"
    seed_records = json.loads(seed_path.read_text(encoding="utf-8"))
    forced_train = {conversation_group(record["conversation"]) for record in seed_records}
    conversations, source_total, rejected = [], 0, 0
    for path in paths:
        print(f"Selecting everyday conversations from {path.name}", flush=True)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                if row["source"] != SOURCE:
                    continue
                source_total += 1
                messages = normalize_and_filter(row["messages"], tokenizer)
                if messages is None:
                    rejected += 1
                else:
                    conversations.append((messages, "smoltalk/everyday"))
    for record in seed_records:
        messages = normalize_and_filter(record["conversation"], tokenizer)
        if messages is None:
            raise ValueError("A curated conversation violates the curriculum limits")
        conversations.append((messages, "authored/chat_basics"))
    random.Random(42).shuffle(conversations)
    records = {name: [] for name in ("train", "validation", "test")}
    groups = {name: set() for name in records}
    seen, omitted_starters = set(), 0
    for messages, source in conversations:
        group = conversation_group(messages)
        bucket = int(group[:8], 16) % 100
        split = "train" if group in forced_train or bucket >= 10 else "validation" if bucket < 5 else "test"
        examples = encode_conversation(messages, tokenizer, args.context, assistant_eos=True)
        # First turns always fit the configured limits. Their common greeting
        # targets are not counted as unseen evaluation answers.
        if split != "train" and has_starter(messages):
            examples = examples[1:]
            omitted_starters += 1
        for example in examples:
            key = hashlib.sha256(example["tokens"].numpy().tobytes() + str(example["answer_start"]).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            records[split].append({**example, "source": source})
            groups[split].add(group)
    if any(len(samples) < 20 for samples in records.values()):
        raise ValueError("Insufficient examples for a separate validation/test split")
    if any(groups[a] & groups[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise ValueError("Conversation group leakage")
    provenance = {"repo": REPO, "revision": REVISION, "source": SOURCE, "license": "Apache-2.0",
                  "files": [{"name": p.name, "sha256": digest_file(p)} for p in paths],
                  "curated_seed": str(seed_path.relative_to(ROOT)), "curated_seed_sha256": digest_file(seed_path),
                  "source_conversations": source_total, "rejected_conversations": rejected,
                  "kept_conversations_including_authored": len(conversations),
                  "evaluation_greeting_targets_omitted": omitted_starters,
                  "filter": "All turns intact; max 40 words/turn, user <=160 tokens, assistant <=128 tokens; no code fences; unknown-token fraction <=1%."}
    for stage, objective in [("dialogue", "dialogue_lm"), ("answers", "assistant_sft")]:
        write_stage(args.output_dir / stage, records, groups, tokenizer_path, provenance, args.context, objective)
    print(f"Prepared clean conversation curriculum: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
