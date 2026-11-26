"""Add conservative Topical-Chat windows to a separate language-stage experiment."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from download_topical_chat import REVISION, SPLITS
from prepare_clean_chat import normalize_and_filter, write_stage
from robbie.core.data.chat_data import digest_file, encode_conversation, load_prepared
from robbie.core.tokenizer.bpe_tokenizer import BPETokenizer


GOOD = {"Good", "Excellent"}
SPLIT_MAP = {"train": "train", "valid_freq": "validation", "valid_rare": "validation",
             "test_freq": "test", "test_rare": "test"}


def personal_only(turn):
    tags = turn.get("knowledge_source")
    return (isinstance(tags, list) and bool(tags)
            and all(str(tag).strip().casefold() == "personal knowledge" for tag in tags))


def selected_windows(conversation, tokenizer, counters):
    """Yield contiguous four-turn windows; never stitch over a rejected turn.

    These are human dialogue language examples, not factual assistant targets.
    Personal Knowledge is a source annotation, not a truth verification.
    """
    ratings = conversation.get("conversation_rating", {})
    content = conversation.get("content", [])
    if set(ratings) != {"agent_1", "agent_2"} or any(value not in GOOD for value in ratings.values()):
        counters["conversation_quality_rejected"] += 1
        return
    if not content or content[0].get("agent") not in {"agent_1", "agent_2"}:
        counters["malformed_conversation"] += 1
        return
    user = content[0]["agent"]
    assistant = "agent_2" if user == "agent_1" else "agent_1"
    if any(turn.get("agent") != (user if i % 2 == 0 else assistant) for i, turn in enumerate(content)):
        counters["nonalternating_conversation_rejected"] += 1
        return
    for end in range(3, len(content), 2):
        window = content[end - 3:end + 1]
        if any(turn.get("turn_rating") not in GOOD for turn in window):
            counters["window_quality_rejected"] += 1
            continue
        if not all(personal_only(turn) for turn in window):
            counters["window_external_or_unknown_grounding_rejected"] += 1
            continue
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": turn["message"]}
                    for i, turn in enumerate(window)]
        normalized = normalize_and_filter(messages, tokenizer)
        if normalized is None:
            counters["window_length_or_text_rejected"] += 1
            continue
        yield normalized, list(range(end - 3, end + 1)), assistant


def fingerprint(sample):
    return hashlib.sha256(sample["tokens"].numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/chat/Topical-Chat" / REVISION)
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "data/processed/clean_chat_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/clean_chat_topical_v1")
    args = parser.parse_args()
    if args.output_dir.resolve() == args.baseline_dir.resolve() or args.output_dir.exists():
        parser.error("Choose a fresh output directory; the baseline and existing experiments are preserved")
    download_path = args.raw_dir / "download_manifest.json"
    if not download_path.is_file():
        parser.error("Download the pinned source first with scripts/download_topical_chat.py")
    download = json.loads(download_path.read_text(encoding="utf-8"))
    if download.get("revision") != REVISION:
        parser.error("Unexpected Topical-Chat revision")
    for item in download["files"]:
        if digest_file(args.raw_dir / item["path"]) != item["sha256"]:
            parser.error(f"Downloaded file changed: {item['path']}")
    tokenizer_path = ROOT / "checkpoints/bpe_stage1_vocab.json"
    tokenizer = BPETokenizer.load(tokenizer_path)
    baseline, language_manifest = load_prepared(args.baseline_dir / "dialogue", tokenizer_path)
    _, answer_manifest = load_prepared(args.baseline_dir / "answers", tokenizer_path)
    if language_manifest["context"] != answer_manifest["context"] or language_manifest.get("dialogue_format") != "assistant-eos-v1":
        parser.error("Incompatible baseline curriculum")
    records = {name: list(samples) for name, samples in baseline.items()}
    groups = {name: set(json.loads((args.baseline_dir / "dialogue" / (name + "_groups.json")).read_text())) for name in records}
    baseline_fingerprints = {fingerprint(sample) for samples in baseline.values() for sample in samples}
    seen_samples, conversation_splits = set(), {}
    source_stats, selected_counts, review_examples = {}, Counter(), []
    # Keep official held-out examples first when removing exact duplicates.
    ordered_splits = ["test_freq", "test_rare", "valid_freq", "valid_rare", "train"]
    for source_split in ordered_splits:
        print(f"Filtering Topical-Chat {source_split}...", flush=True)
        conversations = json.loads((args.raw_dir / "conversations" / (source_split + ".json")).read_text(encoding="utf-8"))
        counters = Counter(raw_conversations=len(conversations))
        split = SPLIT_MAP[source_split]
        for conversation_id, conversation in conversations.items():
            if conversation_id in conversation_splits:
                raise ValueError(f"Conversation ID occurs in multiple source splits: {conversation_id}")
            conversation_splits[conversation_id] = source_split
            for messages, indices, assistant in selected_windows(conversation, tokenizer, counters):
                examples = encode_conversation(messages, tokenizer, language_manifest["context"],
                                               assistant_eos=True, with_turn_index=True)
                sample = next((example for example in examples if example["assistant_turn"] == 3), None)
                if sample is None:
                    counters["window_context_rejected"] += 1
                    continue
                key = fingerprint(sample)
                if key in baseline_fingerprints or key in seen_samples:
                    counters["exact_window_duplicate_rejected"] += 1
                    continue
                seen_samples.add(key)
                group = "topical:" + conversation_id
                records[split].append({**sample, "answer_start": 0, "source": "topical/personal_knowledge_window",
                    "conversation_id": conversation_id, "original_split": source_split,
                    "original_turn_indices": indices, "assistant_agent": assistant,
                    "grounding": "All four turns annotated Personal Knowledge only; not independent factual verification"})
                groups[split].add(group)
                counters["kept_windows"] += 1
                selected_counts[source_split] += 1
                if source_split == "train" and len(review_examples) < 12:
                    review_examples.append({"conversation_id": conversation_id, "turn_indices": indices,
                                            "messages": messages})
        source_stats[source_split] = dict(counters)
    if any(selected_counts[name] < 1 for name in SPLITS):
        raise ValueError(f"A source split has no usable windows: {dict(selected_counts)}")
    if any(groups[a] & groups[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise ValueError("Group leakage in combined curriculum")
    provenance = {"experiment": "clean_chat_topical_v1", "baseline_language_manifest_sha256": digest_file(args.baseline_dir / "dialogue/manifest.json"),
                  "baseline_provenance": language_manifest["provenance"], "topical_download": download,
                  "topical_filter_statistics": source_stats,
                  "topical_usage": "Language adaptation only. Assistant-only reply tuning remains the unchanged clean-chat baseline.",
                  "grounding_policy": "Discard any window with FS/AS/missing grounding tags. Keep only four contiguous Good/Excellent turns annotated Personal Knowledge, in Good/Excellent conversations.",
                  "modifications": "Normalize punctuation, filter short contiguous windows, map the original first speaker to user and other speaker to assistant, add role/EOS tokens, and combine language-stage examples. No turns stitched over gaps."}
    write_stage(args.output_dir / "dialogue", records, groups, tokenizer_path, provenance, language_manifest["context"], "dialogue_lm",
        split_method="Baseline groups unchanged. Topical-Chat official train/valid_freq/valid_rare/test_freq/test_rare splits preserved; grouping by original conversation ID. Exact windows deduplicated with held-out precedence.",
        limits="Human-human language adaptation, not factual assistant training. Personal Knowledge tags are not factual checks; a selected window may still refer to earlier discussion. Reserved test data must not be used for tuning.")
    # Copy baseline reply data byte for byte: no raw human persona claims become
    # assistant-only instruction targets in this experiment.
    reply_dir = args.output_dir / "answers"
    reply_dir.mkdir()
    for name in ("train", "validation", "test"):
        for filename in (name + ".pt", name + "_groups.json"):
            shutil.copyfile(args.baseline_dir / "answers" / filename, reply_dir / filename)
    reply_manifest = {**answer_manifest, "comparison": {
        "baseline_manifest_sha256": digest_file(args.baseline_dir / "answers/manifest.json"),
        "training_inputs_unchanged": True, "experiment": "clean_chat_topical_v1",
        "note": "Same reply training/evaluation examples; initialize from the separately adapted language checkpoint."}}
    (reply_dir / "manifest.json").write_text(json.dumps(reply_manifest, indent=2), encoding="utf-8")
    for source, destination in [("DATALICENSE", "TOPICAL_CHAT_DATALICENSE"), ("src/wiki/NOTICE.md", "TOPICAL_CHAT_WIKI_NOTICE.md"), ("README.md", "TOPICAL_CHAT_ORIGINAL_README.md")]:
        shutil.copyfile(args.raw_dir / source, args.output_dir / destination)
    (args.output_dir / "NOTICE.md").write_text(
        "# Dataset attribution and modification notice\n\nContains modified portions of Amazon Alexa Topical-Chat, revision " + REVISION +
        ". Source: https://github.com/alexa/Topical-Chat . The Topical-Chat portions are provided under CDLA-Sharing-1.0; "
        "the full original agreement is preserved in TOPICAL_CHAT_DATALICENSE. Source credit and the original README are preserved.\n\n"
        "Changes: punctuation normalization, quality/length/grounding filtering, contiguous-window selection, role/EOS formatting, "
        "and combination with the existing baseline language data. The baseline portions retain their source terms in their manifests.\n", encoding="utf-8")
    (args.output_dir / "topical_review_examples.json").write_text(json.dumps(review_examples, indent=2), encoding="utf-8")
    (args.output_dir / "topical_filter_report.json").write_text(json.dumps(source_stats, indent=2), encoding="utf-8")
    print("Topical-Chat windows added:", dict(selected_counts), flush=True)
    print(f"Prepared separate experiment: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
