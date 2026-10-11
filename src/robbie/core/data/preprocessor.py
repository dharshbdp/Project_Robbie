"""Dataset preprocessor and stage pipeline orchestrator for Project Robbie."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from robbie.core.data.cleaner import TextCleaner, DOCUMENT_DELIMITER
from robbie.core.data.sources import fetch_stage1_pretrain_data, fetch_stage2_dialogue_data
from robbie.utils.logger import logger


def calculate_text_stats(text: str) -> Dict[str, Any]:
    """Calculate character, word, line, and estimated token statistics."""
    chars = len(text)
    words = len(text.split())
    lines = len(text.splitlines())
    # Approximation: ~4 characters per token in English
    estimated_tokens = chars // 4
    return {
        "character_count": chars,
        "word_count": words,
        "line_count": lines,
        "estimated_tokens": estimated_tokens,
    }


class DatasetPipeline:
    """Manages acquisition, cleaning, formatting, and splitting for training stages."""

    def __init__(self, data_root: Optional[Path] = None):
        if data_root is None:
            # Default to Project_Robbie/data
            data_root = Path(__file__).resolve().parent.parent.parent.parent / "data"
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"
        self.cleaner = TextCleaner()

    def prepare_stage1_pretrain(self, allow_download: bool = True) -> Dict[str, Any]:
        """Fetch, clean, and process Stage 1 Base Pretraining data (WikiText / Knowledge)."""
        stage1_raw = self.raw_dir / "stage1_pretrain"
        stage1_processed = self.processed_dir / "stage1"
        stage1_processed.mkdir(parents=True, exist_ok=True)

        logger.info("[Stage 1] Fetching raw pretraining data...")
        raw_train_file, raw_val_file = fetch_stage1_pretrain_data(stage1_raw, allow_download=allow_download)

        logger.info("[Stage 1] Cleaning and formatting text...")
        with open(raw_train_file, "r", encoding="utf-8", errors="replace") as f:
            raw_train_text = f.read()

        with open(raw_val_file, "r", encoding="utf-8", errors="replace") as f:
            raw_val_text = f.read()

        cleaned_train = self.cleaner.clean(raw_train_text)
        cleaned_val = self.cleaner.clean(raw_val_text)

        # Write clean files to processed/stage1
        proc_train_path = stage1_processed / "train.txt"
        proc_val_path = stage1_processed / "val.txt"

        with open(proc_train_path, "w", encoding="utf-8") as f:
            f.write(cleaned_train)
        with open(proc_val_path, "w", encoding="utf-8") as f:
            f.write(cleaned_val)

        train_stats = calculate_text_stats(cleaned_train)
        val_stats = calculate_text_stats(cleaned_val)

        metadata = {
            "stage": 1,
            "name": "Base Model Pretraining",
            "description": "General English patterns, sentence structure, and encyclopedic knowledge",
            "source_files": [str(raw_train_file.name), str(raw_val_file.name)],
            "train": train_stats,
            "val": val_stats,
            "delimiter": DOCUMENT_DELIMITER,
        }

        meta_path = stage1_processed / "dataset_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[Stage 1] Preparation complete! Train: {train_stats['character_count']:,} chars, Val: {val_stats['character_count']:,} chars")
        return metadata

    def prepare_stage2_dialogue(self, train_ratio: float = 0.85, force: bool = False) -> Dict[str, Any]:
        """Format, clean, and split Stage 2 Conversational & Assistant instruction data."""
        stage2_raw = self.raw_dir / "stage2_dialogue"
        stage2_processed = self.processed_dir / "stage2"
        stage2_processed.mkdir(parents=True, exist_ok=True)

        logger.info("[Stage 2] Fetching conversational dialogue data...")
        raw_dialogue_file = fetch_stage2_dialogue_data(stage2_raw, force=force)

        # Parse JSONL conversations
        conversations: List[str] = []
        with open(raw_dialogue_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    conv_lines = []
                    for turn in data.get("conversation", []):
                        role = "User" if turn["role"] == "user" else "Robbie"
                        content = turn["content"].strip()
                        conv_lines.append(f"{role}: {content}")
                    if conv_lines:
                        # Append conversation with boundary delimiter
                        dialogue_block = "\n".join(conv_lines)
                        conversations.append(dialogue_block)
                except Exception as e:
                    logger.warning(f"Error parsing dialogue line: {e}")

        # Split at conversation level (zero conversation leakage between train and val)
        split_idx = max(1, int(len(conversations) * train_ratio))
        train_convs = conversations[:split_idx]
        val_convs = conversations[split_idx:] if split_idx < len(conversations) else conversations[-1:]

        delimiter = f"\n\n{DOCUMENT_DELIMITER}\n\n"
        cleaned_train = delimiter.join([self.cleaner.clean(c) for c in train_convs])
        cleaned_val = delimiter.join([self.cleaner.clean(c) for c in val_convs])

        proc_train_path = stage2_processed / "train.txt"
        proc_val_path = stage2_processed / "val.txt"

        with open(proc_train_path, "w", encoding="utf-8") as f:
            f.write(cleaned_train)
        with open(proc_val_path, "w", encoding="utf-8") as f:
            f.write(cleaned_val)

        train_stats = calculate_text_stats(cleaned_train)
        val_stats = calculate_text_stats(cleaned_val)

        metadata = {
            "stage": 2,
            "name": "Conversational Fine-Tuning",
            "description": "Dialogue turns, assistant persona, instructions, and query-response formatting",
            "conversations_total": len(conversations),
            "conversations_train": len(train_convs),
            "conversations_val": len(val_convs),
            "train": train_stats,
            "val": val_stats,
            "delimiter": DOCUMENT_DELIMITER,
        }

        meta_path = stage2_processed / "dataset_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[Stage 2] Preparation complete! Dialogues: {len(conversations)} | Train: {train_stats['character_count']:,} chars, Val: {val_stats['character_count']:,} chars")
        return metadata
