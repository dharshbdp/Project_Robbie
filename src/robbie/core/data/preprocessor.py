"""Dataset preprocessor and stage pipeline orchestrator for Project Robbie."""

import csv
import json
from pathlib import Path
import re
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
            from robbie.config import DATA_DIR
            data_root = DATA_DIR
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

        source_files = [str(raw_train_file.name), str(raw_val_file.name)]

        # Check for additional pretraining corpora (e.g. TinyStories)
        tinystories_path = stage1_raw / "tinystories.txt"
        if tinystories_path.exists() and tinystories_path.stat().st_size > 0:
            logger.info(f"[Stage 1] Ingesting additional narrative corpus: {tinystories_path.name}")
            with open(tinystories_path, "r", encoding="utf-8", errors="replace") as f:
                tinystories_text = f.read()
            raw_train_text = raw_train_text + f"\n\n{DOCUMENT_DELIMITER}\n\n" + tinystories_text
            source_files.append(tinystories_path.name)

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
            "description": "General English patterns, sentence structure, and encyclopedic knowledge + narrative stories",
            "source_files": source_files,
            "train": train_stats,
            "val": val_stats,
            "delimiter": DOCUMENT_DELIMITER,
        }

        meta_path = stage1_processed / "dataset_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[Stage 1] Preparation complete! Train: {train_stats['character_count']:,} chars, Val: {val_stats['character_count']:,} chars")
        return metadata

    def _extract_dialogue_block(self, item: Dict[str, Any]) -> Optional[str]:
        """Convert an arbitrary dictionary into a standardized 'User: ...\nRobbie: ...' dialogue block."""
        if not isinstance(item, dict):
            return None

        # 1. Standard conversation format: {"conversation": [{"role": "user", "content": "..."}, ...]}
        if "conversation" in item and isinstance(item["conversation"], list):
            conv_lines = []
            for turn in item["conversation"]:
                role_val = str(turn.get("role", "")).lower()
                role = "User" if role_val in ("user", "human") else "Robbie"
                content = str(turn.get("content", "")).strip()
                if content:
                    conv_lines.append(f"{role}: {content}")
            if len(conv_lines) >= 2:
                return "\n".join(conv_lines)

        # 2. Alpaca/Instruction format: {"instruction": "...", "input": "...", "output": "..."}
        if "instruction" in item and "output" in item:
            inst = str(item["instruction"]).strip()
            inp = str(item.get("input", "")).strip()
            out = str(item["output"]).strip()
            user_msg = f"{inst}\n{inp}".strip() if inp else inst
            if user_msg and out:
                return f"User: {user_msg}\nRobbie: {out}"

        # 3. Q&A format: {"question": "...", "answer": "..."}
        if "question" in item and "answer" in item:
            q = str(item["question"]).strip()
            a = str(item["answer"]).strip()
            if q and a:
                return f"User: {q}\nRobbie: {a}"

        # 4. Prompt/Response format: {"prompt": "...", "response": "..."}
        if "prompt" in item and "response" in item:
            p = str(item["prompt"]).strip()
            r = str(item["response"]).strip()
            if p and r:
                return f"User: {p}\nRobbie: {r}"

        # 5. User/Assistant key pair format: {"user": "...", "assistant": "..."}
        if "user" in item and ("assistant" in item or "robbie" in item):
            u = str(item["user"]).strip()
            resp = str(item.get("assistant") or item.get("robbie")).strip()
            if u and resp:
                return f"User: {u}\nRobbie: {resp}"

        return None

    def _parse_dialogue_file(self, file_path: Path) -> List[str]:
        """Parse any JSONL, JSON, CSV, or TXT file into formatted dialogue blocks."""
        ext = file_path.suffix.lower()
        dialogues = []

        if ext == ".jsonl":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        block = self._extract_dialogue_block(data)
                        if block:
                            dialogues.append(block)
                    except Exception:
                        continue

        elif ext == ".json":
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        block = self._extract_dialogue_block(item)
                        if block:
                            dialogues.append(block)
            except Exception as e:
                logger.warning(f"Error reading JSON file {file_path.name}: {e}")

        elif ext == ".csv":
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        block = self._extract_dialogue_block(row)
                        if block:
                            dialogues.append(block)
            except Exception as e:
                logger.warning(f"Error reading CSV file {file_path.name}: {e}")

        elif ext == ".txt":
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                sections = re.split(r"\n\s*\n", text)
                for s in sections:
                    s_clean = s.strip()
                    if ("User:" in s_clean or "user:" in s_clean) and ("Robbie:" in s_clean or "Assistant:" in s_clean or "assistant:" in s_clean):
                        dialogues.append(s_clean)
            except Exception as e:
                logger.warning(f"Error reading TXT file {file_path.name}: {e}")

        return dialogues

    def prepare_stage2_dialogue(self, train_ratio: float = 0.85, force: bool = False) -> Dict[str, Any]:
        """Format, clean, and split Stage 2 Conversational & Assistant instruction data."""
        stage2_raw = self.raw_dir / "stage2_dialogue"
        stage2_processed = self.processed_dir / "stage2"
        stage2_processed.mkdir(parents=True, exist_ok=True)

        logger.info("[Stage 2] Loading conversational dialogue data from core, open, and custom sources...")
        raw_dialogue_file = fetch_stage2_dialogue_data(stage2_raw, force=force)

        all_conversations: List[str] = []
        source_counts: Dict[str, int] = {}

        # 1. Load core assistant dialogues (Robbie personality, memory, awareness)
        core_dialogues = self._parse_dialogue_file(raw_dialogue_file)
        all_conversations.extend(core_dialogues)
        source_counts[raw_dialogue_file.name] = len(core_dialogues)

        # 2. Discover any additional datasets in stage2_raw (e.g. downloaded open dialogues)
        for p in sorted(stage2_raw.glob("*.jsonl")):
            if p.name != raw_dialogue_file.name:
                dl_dialogues = self._parse_dialogue_file(p)
                if dl_dialogues:
                    all_conversations.extend(dl_dialogues)
                    source_counts[p.name] = len(dl_dialogues)

        # 3. Discover user-provided custom datasets in data/raw/custom/
        custom_dir = self.raw_dir / "custom"
        if custom_dir.exists():
            for p in sorted(custom_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in (".jsonl", ".json", ".csv", ".txt") and p.name != "README.md":
                    custom_dialogues = self._parse_dialogue_file(p)
                    if custom_dialogues:
                        all_conversations.extend(custom_dialogues)
                        source_counts[f"custom/{p.name}"] = len(custom_dialogues)

        for src, cnt in source_counts.items():
            logger.info(f"[Stage 2 Source] {src}: {cnt:,} dialogues loaded")

        # Deterministically shuffle conversations so topics are balanced across splits
        import random
        rng = random.Random(42)
        shuffled = list(all_conversations)
        rng.shuffle(shuffled)

        # Split at conversation level (zero conversation leakage between train and val)
        split_idx = max(1, int(len(shuffled) * train_ratio))
        train_convs = shuffled[:split_idx]
        val_convs = shuffled[split_idx:] if split_idx < len(shuffled) else shuffled[-1:]

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
            "conversations_total": len(all_conversations),
            "conversations_train": len(train_convs),
            "conversations_val": len(val_convs),
            "sources": source_counts,
            "train": train_stats,
            "val": val_stats,
            "delimiter": DOCUMENT_DELIMITER,
        }

        meta_path = stage2_processed / "dataset_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[Stage 2] Preparation complete! Total Dialogues: {len(all_conversations):,} | Sources: {len(source_counts)} | Train: {train_stats['character_count']:,} chars, Val: {val_stats['character_count']:,} chars")
        return metadata
