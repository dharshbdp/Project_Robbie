"""Tests for TinyLLM model architecture, training loop, and generation."""

import pytest
import torch
import tempfile
from pathlib import Path

from robbie.core.models.tiny_llm import TinyLLM, TinyLLMConfig
from robbie.core.training.trainer import Trainer, TrainingConfig
from robbie.core.tokenizer import TextDataset


class TestTinyLLM:
    """Test suite for TinyLLM neural model."""

    def test_forward_pass_shapes(self):
        config = TinyLLMConfig(vocab_size=50, block_size=16, n_embd=32, n_hidden=64)
        model = TinyLLM(config)

        # Batch size 2, sequence length 8
        x = torch.randint(0, config.vocab_size, (2, 8))
        logits, loss = model(x)

        assert logits.shape == (2, 8, config.vocab_size)
        assert loss is None

    def test_forward_loss_calculation(self):
        config = TinyLLMConfig(vocab_size=50, block_size=16, n_embd=32, n_hidden=64)
        model = TinyLLM(config)

        x = torch.randint(0, config.vocab_size, (2, 8))
        y = torch.randint(0, config.vocab_size, (2, 8))
        logits, loss = model(x, targets=y)

        assert loss is not None
        assert isinstance(loss.item(), float)
        assert loss.item() > 0.0

    def test_exceeding_block_size_raises(self):
        config = TinyLLMConfig(vocab_size=50, block_size=8)
        model = TinyLLM(config)

        x = torch.randint(0, config.vocab_size, (1, 10))
        with pytest.raises(ValueError, match="exceeds model block size"):
            model(x)

    def test_autoregressive_generation(self):
        config = TinyLLMConfig(vocab_size=40, block_size=16, n_embd=32, n_hidden=64)
        model = TinyLLM(config)

        # Prompt of length 3
        prompt = torch.tensor([[1, 5, 10]])
        output = model.generate(prompt, max_new_tokens=5, temperature=1.0)

        assert output.shape == (1, 8)  # 3 prompt tokens + 5 generated tokens

    def test_greedy_generation(self):
        config = TinyLLMConfig(vocab_size=40, block_size=16, n_embd=32, n_hidden=64)
        model = TinyLLM(config)

        prompt = torch.tensor([[2, 4]])
        output = model.generate(prompt, max_new_tokens=4, temperature=0.0)

        assert output.shape == (1, 6)


class TestTrainingAndCheckpointing:
    """Test suite for training loop and checkpoint saving/loading."""

    def test_training_loss_decreases(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_path = Path(tmp_dir)
            tokens = [1, 2, 3, 4] * 50
            dataset = TextDataset(tokens, seq_len=8)

            config = TinyLLMConfig(vocab_size=10, block_size=16, n_embd=32, n_hidden=64)
            model = TinyLLM(config)

            train_cfg = TrainingConfig(
                epochs=3,
                batch_size=8,
                learning_rate=1e-2,
                checkpoint_dir=str(tmp_path),
                device="cpu",
            )

            trainer = Trainer(model, train_dataset=dataset, val_dataset=dataset, config=train_cfg)
            history = trainer.fit()

            assert len(history["train_loss"]) == 3
            assert history["train_loss"][-1] < history["train_loss"][0]

    def test_checkpoint_save_and_load(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = TinyLLMConfig(vocab_size=20, block_size=16, n_embd=32, n_hidden=64)
            model1 = TinyLLM(config)

            checkpoint_file = tmp_path / "test_model.pt"
            tokens = [1, 2, 3] * 30
            dataset = TextDataset(tokens, seq_len=4)

            trainer = Trainer(
                model1,
                train_dataset=dataset,
                val_dataset=dataset,
                config=TrainingConfig(epochs=1, batch_size=4, checkpoint_dir=str(tmp_path), device="cpu"),
            )
            trainer.save_checkpoint(checkpoint_file, epoch=1, val_loss=1.23)
            assert checkpoint_file.exists()

            model2 = TinyLLM(config)
            Trainer.load_checkpoint(checkpoint_file, model2)

            model1.eval()
            model2.eval()
            test_input = torch.tensor([[1, 2, 3]])
            with torch.no_grad():
                out1, _ = model1(test_input)
                out2, _ = model2(test_input)

            assert torch.allclose(out1, out2)
