"""Unit tests for CausalSelfAttention, TransformerBlock, and RobbieTransformer."""

import pytest
import torch
import tempfile
from pathlib import Path

from robbie.core.models.attention import CausalSelfAttention
from robbie.core.models.transformer_block import TransformerBlock, MLP
from robbie.core.models.transformer import RobbieTransformer, TransformerConfig
from robbie.core.training.trainer import Trainer, TrainingConfig
from robbie.core.tokenizer import TextDataset


class TestAttention:
    """Test suite for CausalSelfAttention."""

    def test_attention_shapes(self):
        attn = CausalSelfAttention(n_embd=64, n_head=4, block_size=16)
        x = torch.randn(2, 8, 64)  # (Batch=2, Seq=8, Emb=64)
        out = attn(x)
        assert out.shape == (2, 8, 64)

    def test_causal_mask_structure(self):
        attn = CausalSelfAttention(n_embd=32, n_head=2, block_size=8)
        mask = attn.bias.squeeze()
        # Verify lower triangular: diagonal and below is 1, above diagonal is 0
        assert torch.all(torch.tril(mask) == mask)
        assert mask[0, 1] == 0.0
        assert mask[1, 0] == 1.0


class TestTransformerBlock:
    """Test suite for TransformerBlock."""

    def test_block_shape_and_residual(self):
        block = TransformerBlock(n_embd=64, n_head=4, block_size=16)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)


class TestRobbieTransformer:
    """Test suite for full RobbieTransformer architecture."""

    def test_transformer_forward_pass(self):
        config = TransformerConfig(
            vocab_size=50,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.1,
        )
        model = RobbieTransformer(config)

        x = torch.randint(0, config.vocab_size, (2, 8))
        logits, loss = model(x)

        assert logits.shape == (2, 8, config.vocab_size)
        assert loss is None

    def test_transformer_loss_computation(self):
        config = TransformerConfig(
            vocab_size=50,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=32,
        )
        model = RobbieTransformer(config)

        x = torch.randint(0, config.vocab_size, (2, 8))
        y = torch.randint(0, config.vocab_size, (2, 8))
        logits, loss = model(x, targets=y)

        assert loss is not None
        assert loss.item() > 0.0

    def test_transformer_generation(self):
        config = TransformerConfig(
            vocab_size=40,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=32,
        )
        model = RobbieTransformer(config)

        prompt = torch.tensor([[1, 2, 3]])
        out = model.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=20)
        assert out.shape == (1, 8)  # 3 initial tokens + 5 generated tokens

    def test_transformer_training_and_checkpoint(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_path = Path(tmp_dir)
            tokens = [1, 2, 3, 4] * 40
            dataset = TextDataset(tokens, seq_len=8)

            config = TransformerConfig(
                vocab_size=10,
                block_size=16,
                n_layer=2,
                n_head=2,
                n_embd=32,
            )
            model1 = RobbieTransformer(config)
            checkpoint_file = tmp_path / "transformer_test.pt"

            trainer = Trainer(
                model1,
                train_dataset=dataset,
                val_dataset=dataset,
                config=TrainingConfig(epochs=1, batch_size=4, checkpoint_dir=str(tmp_path), device="cpu"),
            )
            trainer.save_checkpoint(checkpoint_file, epoch=1, val_loss=1.0)
            assert checkpoint_file.exists()

            model2 = RobbieTransformer(config)
            Trainer.load_checkpoint(checkpoint_file, model2)

            model1.eval()
            model2.eval()
            test_input = torch.tensor([[1, 2, 3]])
            with torch.no_grad():
                out1, _ = model1(test_input)
                out2, _ = model2(test_input)

            assert torch.allclose(out1, out2)

    def test_early_stopping_and_best_checkpoint(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_path = Path(tmp_dir)
            tokens = [1, 2, 3, 4] * 30
            train_ds = TextDataset(tokens, seq_len=8)
            val_ds = TextDataset(tokens, seq_len=8)

            config = TransformerConfig(vocab_size=10, block_size=16, n_layer=1, n_head=1, n_embd=16)
            model = RobbieTransformer(config)

            train_cfg = TrainingConfig(
                epochs=10,
                batch_size=4,
                checkpoint_dir=str(tmp_path),
                best_model_name="best_custom.pt",
                early_stopping_patience=2,
                device="cpu",
            )
            trainer = Trainer(model, train_ds, val_ds, train_cfg)
            history = trainer.fit()

            assert (tmp_path / "best_custom.pt").exists()
            assert (tmp_path / "latest_model.pt").exists()
            assert "best_epoch" in history
            assert "best_val_loss" in history

    def test_checkpoint_metadata_and_reproducible_generation(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = TransformerConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32)
            model_orig = RobbieTransformer(config)
            ckpt_path = tmp_path / "test_repro.pt"

            trainer = Trainer(
                model_orig,
                train_dataset=None,
                val_dataset=None,
                config=TrainingConfig(checkpoint_dir=str(tmp_path), device="cpu"),
            )
            trainer.save_checkpoint(ckpt_path, epoch=7, val_loss=1.4713)

            # Load into fresh model
            model_loaded = RobbieTransformer(config)
            metadata = Trainer.load_checkpoint(ckpt_path, model_loaded)

            assert metadata["epoch"] == 7
            assert abs(metadata["val_loss"] - 1.4713) < 1e-4

            # Verify identical outputs and reproducible generation
            model_orig.eval()
            model_loaded.eval()
            inp = torch.tensor([[5, 6, 7]])

            torch.manual_seed(99)
            gen1 = model_orig.generate(inp, max_new_tokens=6, temperature=0.8, top_k=5)

            torch.manual_seed(99)
            gen2 = model_loaded.generate(inp, max_new_tokens=6, temperature=0.8, top_k=5)

            assert torch.equal(gen1, gen2)
