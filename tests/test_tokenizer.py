"""Comprehensive tests for Robbie tokenizers and dataset preparation."""

import pytest
import tempfile
from pathlib import Path

from robbie.core.tokenizer import (
    CharTokenizer,
    BPETokenizer,
    SpecialTokens,
    TextDataset,
    split_text_data,
    prepare_datasets,
    prepare_datasets_from_splits,
)


class TestCharTokenizer:
    """Test suite for CharTokenizer."""

    def test_train_and_vocab_size(self):
        tokenizer = CharTokenizer()
        text = "hello world"
        tokenizer.train([text])

        # 4 special tokens + unique chars in 'hello world' (' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w') = 4 + 8 = 12
        assert tokenizer.vocab_size == 12
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 1
        assert tokenizer.bos_token_id == 2
        assert tokenizer.eos_token_id == 3

    def test_encode_and_decode_roundtrip(self):
        tokenizer = CharTokenizer()
        sample = "Robbie is learning AI from the ground up!"
        tokenizer.train([sample])

        encoded = tokenizer.encode(sample)
        assert isinstance(encoded, list)
        assert all(isinstance(i, int) for i in encoded)

        decoded = tokenizer.decode(encoded)
        assert decoded == sample

    def test_special_tokens_handling(self):
        tokenizer = CharTokenizer()
        tokenizer.train(["abc"])

        encoded = tokenizer.encode("abc", add_special_tokens=True)
        assert encoded[0] == tokenizer.bos_token_id
        assert encoded[-1] == tokenizer.eos_token_id

        # Decoding with skip_special_tokens=True
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        assert decoded == "abc"

    def test_unknown_character(self):
        tokenizer = CharTokenizer()
        tokenizer.train(["abc"])

        encoded = tokenizer.encode("abz")
        assert encoded[-1] == tokenizer.unk_token_id

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_path = Path(tmp_dir)
            tokenizer = CharTokenizer()
            sample = "Persist this vocabulary!"
            tokenizer.train([sample])

            save_path = tmp_path / "char_vocab.json"
            tokenizer.save(save_path)
            assert save_path.exists()

            loaded = CharTokenizer.load(save_path)
            assert loaded.vocab_size == tokenizer.vocab_size
            assert loaded.encode(sample) == tokenizer.encode(sample)
            assert loaded.decode(loaded.encode(sample)) == sample


class TestBPETokenizer:
    """Test suite for BPETokenizer."""

    def test_bpe_training_and_merges(self):
        tokenizer = BPETokenizer(target_vocab_size=30)
        corpus = [
            "low low low low low",
            "lower lower",
            "newest newest newest newest",
            "widest widest",
        ]
        tokenizer.train(corpus)

        assert tokenizer.vocab_size > len(tokenizer.special_tokens.all_tokens)
        assert len(tokenizer.merges) > 0

    def test_bpe_encode_decode_roundtrip(self):
        tokenizer = BPETokenizer(target_vocab_size=50)
        corpus = [
            "Robbie is learning to think and speak.",
            "Robbie remembers previous conversations.",
        ]
        tokenizer.train(corpus)

        text = "Robbie is learning to think"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_bpe_special_tokens(self):
        tokenizer = BPETokenizer(target_vocab_size=40)
        tokenizer.train(["hello world"])

        encoded = tokenizer.encode("hello", add_special_tokens=True)
        assert encoded[0] == tokenizer.bos_token_id
        assert encoded[-1] == tokenizer.eos_token_id

        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        assert decoded == "hello"

    def test_bpe_save_and_load(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_path = Path(tmp_dir)
            tokenizer = BPETokenizer(target_vocab_size=50)
            corpus = ["testing bpe persistence for robbie"]
            tokenizer.train(corpus)

            save_path = tmp_path / "bpe_vocab.json"
            tokenizer.save(save_path)
            assert save_path.exists()

            loaded = BPETokenizer.load(save_path)
            assert loaded.vocab_size == tokenizer.vocab_size
            assert loaded.merges == tokenizer.merges

            sample = "testing bpe persistence"
            assert loaded.encode(sample) == tokenizer.encode(sample)
            assert loaded.decode(loaded.encode(sample)) == sample


class TestDatasetAndPreprocessing:
    """Test suite for text dataset creation and preprocessing."""

    def test_split_text_data(self):
        text = "0123456789"
        train, val = split_text_data(text, train_ratio=0.8)
        assert train == "01234567"
        assert val == "89"

    def test_text_dataset_sequence_alignment(self):
        tokens = list(range(20))
        dataset = TextDataset(tokens, seq_len=5)

        assert len(dataset) == 15
        x, y = dataset[0]
        # x is tokens[0:5], y is tokens[1:6] (next token prediction)
        assert x.tolist() == [0, 1, 2, 3, 4]
        assert y.tolist() == [1, 2, 3, 4, 5]

    def test_prepare_datasets(self):
        text = "The quick brown fox jumps over the lazy dog. Robbie is a personal AI system."
        tokenizer = CharTokenizer()
        tokenizer.train([text])

        train_ds, val_ds = prepare_datasets(text, tokenizer, seq_len=10, train_ratio=0.8)
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        x, y = train_ds[0]
        assert x.shape == (10,)
        assert y.shape == (10,)

    def test_prepare_datasets_from_splits(self):
        train_text = "This is the training set for Robbie base pretraining."
        val_text = "This is validation."
        tokenizer = CharTokenizer()
        tokenizer.train([train_text, val_text])

        train_ds, val_ds = prepare_datasets_from_splits(
            train_text, val_text, tokenizer, seq_len=8
        )
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        x, y = train_ds[0]
        assert x.shape == (8,)
        assert y.shape == (8,)

