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
