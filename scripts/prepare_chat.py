"""Download pinned public conversations and prepare complete-answer SFT data."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow.parquet as pq
import torch

from robbie.core.data.chat_data import (
    FORMAT_VERSION, digest_file, encode_conversation, prompt_group, split_for_group,
)
from robbie.core.tokenizer.bpe_tokenizer import BPETokenizer


SOURCES = [
    {"repo": "HuggingFaceTB/smol-smoltalk", "revision": "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
     "license": "Apache-2.0", "files": [f"data/train-{i:05d}-of-00004.parquet" for i in range(4)]},
    {"repo": "OpenAssistant/oasst1", "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
     "license": "Apache-2.0", "files": ["data/train-00000-of-00001-b42a775f407cee45.parquet"]},
]


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Robbie-training/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as out:
                size = int(response.headers.get("Content-Length", 0))
                total, last = 0, time.monotonic()
                while block := response.read(1024 * 1024):
                    out.write(block)
                    total += len(block)
                    if time.monotonic() - last > 5:
                        print(f"  {path.name}: {total / 1e6:.0f}/{size / 1e6:.0f} MB", flush=True)
                        last = time.monotonic()
            pq.read_metadata(temporary)  # Reject partial or invalid files before caching.
            os.replace(temporary, path)
            return
        except Exception:
            if attempt == 2:
                raise
            print(f"Download interrupted; retry {attempt + 1}/2", flush=True)
            time.sleep(2)


def smol_records(paths, maximum, seed):
    """Uniform reservoir across all shards, before context-length filtering."""
    rng, reservoir, seen = random.Random(seed), [], 0
    for path in paths:
        print(f"Reading {path.name}", flush=True)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            for record in batch.to_pylist():
                messages = record["messages"]
                if not valid_messages(messages):
                    continue
                item = {"messages": messages, "source": "smol/" + record["source"]}
                seen += 1
                if len(reservoir) < maximum:
                    reservoir.append(item)
                else:
                    index = rng.randrange(seen)
                    if index < maximum:
                        reservoir[index] = item
    rng.shuffle(reservoir)
    print(f"Sampled {len(reservoir):,} of {seen:,} SmolTalk conversations across all shards", flush=True)
    return reservoir


def valid_messages(messages):
    roles = [m["role"] for m in messages]
    dialogue = roles[1:] if roles and roles[0] == "system" else roles
    return (len(dialogue) >= 2 and len(dialogue) % 2 == 0
            and all(role == ("user" if i % 2 == 0 else "assistant")
                    for i, role in enumerate(dialogue))
            and all(isinstance(m["content"], str) and m["content"].strip() for m in messages))


def oasst_records(path):
    rows = pq.read_table(path).to_pylist()
    by_id = {row["message_id"]: row for row in rows}
    records = []
    for row in rows:
        if row["role"] != "assistant" or row["rank"] != 0:
            continue
        chain, current = [], row
        while current is not None:
            if (current["lang"] != "en" or current["deleted"]
                    or not current["review_result"] or current.get("synthetic")):
                break
            chain.append({"role": "user" if current["role"] == "prompter" else "assistant",
                          "content": current["text"]})
            if current["parent_id"] is None:
                messages = list(reversed(chain))
                if valid_messages(messages):
                    records.append({"messages": messages, "source": "oasst1/human"})
                break
            current = by_id.get(current["parent_id"])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/chat_v1")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--max-conversations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.context < 32 or args.max_conversations < 100:
        parser.error("context must be >=32 and max-conversations >=100")
    if (args.output_dir / "manifest.json").exists():
        parser.error("Prepared data already exists. Choose another output directory to preserve active runs.")
    paths_by_source, downloads = [], []
    for source in SOURCES:
        paths = []
        for filename in source["files"]:
            path = ROOT / "data/raw/chat" / source["repo"].replace("/", "_") / source["revision"] / Path(filename).name
            print(f"Dataset: {source['repo']} / {Path(filename).name}", flush=True)
            download(f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{filename}", path)
            downloads.append({"repo": source["repo"], "revision": source["revision"],
                              "file": filename, "sha256": digest_file(path), "license": source["license"]})
            paths.append(path)
        paths_by_source.append(paths)
    records = smol_records(paths_by_source[0], args.max_conversations, args.seed)
    records.extend(oasst_records(paths_by_source[1][0]))
    random.Random(args.seed).shuffle(records)
    tokenizer_path = ROOT / "checkpoints/bpe_stage1_vocab.json"
    tokenizer = BPETokenizer.load(tokenizer_path)
    splits = {name: [] for name in ("train", "validation", "test")}
    groups = {name: set() for name in splits}
    counts = {name: Counter() for name in splits}
    seen, skipped, duplicates = set(), 0, 0
    for index, record in enumerate(records):
        messages = record["messages"]
        group = prompt_group(messages)
        split = split_for_group(group)
        examples = encode_conversation(messages, tokenizer, args.context)
        if not examples:
            skipped += 1
        for example in examples:
            # Deduplicate repeated ancestor answers in OpenAssistant tree paths,
            # as well as exact overlapping samples from other sources.
            key = example["tokens"].numpy().tobytes() + str(example["answer_start"]).encode()
            fingerprint = hashlib.sha256(key).hexdigest()
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            splits[split].append(example)
            groups[split].add(group)
            counts[split][record["source"]] += 1
        if (index + 1) % 2000 == 0:
            print(f"Tokenized {index + 1:,}/{len(records):,} conversations; "
                  f"{sum(map(len, splits.values())):,} complete answers fit", flush=True)
    if any(len(samples) < 10 for samples in splits.values()):
        raise ValueError("Too few examples in a split. Increase conversation count or context length.")
    assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"]
                or groups["validation"] & groups["test"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": FORMAT_VERSION, "context": args.context, "seed": args.seed,
                "tokenizer_sha256": digest_file(tokenizer_path), "downloads": downloads,
                "sampled_conversations": len(records), "no_fitting_answers": skipped,
                "duplicate_examples_removed": duplicates, "splits": {},
                "split_method": "SHA256 of normalized opening user prompt; 97/2/1 percent",
                "limits": "Exact prompt grouping is not semantic deduplication. Short-answer English SFT; not live computer/tool training."}
    for name, samples in splits.items():
        path = args.output_dir / (name + ".pt")
        torch.save(samples, path)
        manifest["splits"][name] = {
            "examples": len(samples), "groups": len(groups[name]), "sha256": digest_file(path),
            "answer_tokens": sum(len(s["tokens"]) - 1 - s["answer_start"] for s in samples),
            "input_tokens": sum(len(s["tokens"]) - 1 for s in samples), "sources": dict(counts[name]),
        }
        (args.output_dir / (name + "_groups.json")).write_text(json.dumps(sorted(groups[name])), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["splits"], indent=2), flush=True)
    print(f"Prepared data: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
