"""Unit tests for Project Robbie dataset cleaning, sources, and pipeline."""

import json
from pathlib import Path
import tempfile
import pytest

from robbie.core.data.cleaner import TextCleaner, DOCUMENT_DELIMITER
from robbie.core.data.preprocessor import DatasetPipeline, calculate_text_stats


class TestTextCleaner:
    def test_unicode_and_quotes_normalization(self):
        cleaner = TextCleaner()
        dirty = "“Hello world,” said the ‘AI’—with style."
        cleaned = cleaner.clean(dirty)
        assert cleaned == '"Hello world," said the \'AI\' - with style.'

    def test_whitespace_and_control_characters(self):
        cleaner = TextCleaner()
        dirty = "Line 1   \n\n\n\n\nLine 2\x00\x08 with bad chars   "
        cleaned = cleaner.clean(dirty)
        assert cleaned == "Line 1\n\nLine 2 with bad chars"

    def test_document_joining(self):
        cleaner = TextCleaner()
        docs = ["Document one.", "Document two."]
        joined = cleaner.join_documents(docs)
        assert DOCUMENT_DELIMITER in joined
        assert "Document one." in joined
        assert "Document two." in joined


class TestDatasetPipeline:
    def test_calculate_text_stats(self):
        text = "Hello world from Robbie AI."
        stats = calculate_text_stats(text)
        assert stats["character_count"] == len(text)
        assert stats["word_count"] == 5
        assert stats["estimated_tokens"] == len(text) // 4

    def test_stage1_pipeline_execution(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_root = Path(tmp_dir)
            pipeline = DatasetPipeline(data_root=tmp_root)
            meta = pipeline.prepare_stage1_pretrain(allow_download=False)

            assert meta["stage"] == 1
            assert (tmp_root / "processed" / "stage1" / "train.txt").exists()
            assert (tmp_root / "processed" / "stage1" / "val.txt").exists()
            assert (tmp_root / "processed" / "stage1" / "dataset_metadata.json").exists()

            # Verify content is non-empty
            train_content = (tmp_root / "processed" / "stage1" / "train.txt").read_text(encoding="utf-8")
            val_content = (tmp_root / "processed" / "stage1" / "val.txt").read_text(encoding="utf-8")
            assert len(train_content) > 100
            assert len(val_content) > 50

    def test_stage2_pipeline_execution(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_root = Path(tmp_dir)
            pipeline = DatasetPipeline(data_root=tmp_root)
            meta = pipeline.prepare_stage2_dialogue(train_ratio=0.8)

            assert meta["stage"] == 2
            assert meta["conversations_total"] > 0
            assert (tmp_root / "processed" / "stage2" / "train.txt").exists()
            assert (tmp_root / "processed" / "stage2" / "val.txt").exists()
            assert (tmp_root / "processed" / "stage2" / "dataset_metadata.json").exists()

            # Verify dialogue format: User / Robbie turns present
            train_content = (tmp_root / "processed" / "stage2" / "train.txt").read_text(encoding="utf-8")
            assert "User:" in train_content
            assert "Robbie:" in train_content
            assert DOCUMENT_DELIMITER in train_content
