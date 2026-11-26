"""Unit tests for Robbie-300M parameter architecture and VRAM safety guard."""

import pytest
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robbie.core.models.transformer import RobbieTransformer, TransformerConfig
from scripts.train_300m_chat import SafeVRAMGuard
from scripts.prepare_300m_curriculum import normalize_text


def test_300m_parameter_count():
    """Verify Robbie-300M architecture has ~304.8M parameters."""
    config = TransformerConfig(
        vocab_size=2048,
        block_size=512,
        n_layer=24,
        n_head=16,
        n_embd=1024,
        dropout=0.05,
    )
    model = RobbieTransformer(config)
    param_count = sum(p.numel() for p in model.parameters())
    assert 300_000_000 < param_count < 310_000_000, f"Expected ~304.8M, got {param_count:,}"
    assert param_count == 304_834_560


def test_text_normalization():
    """Verify unicode quotes, dashes, and replacement characters are normalized cleanly."""
    dirty = "What\u2019s the best way \u201cFix Drain\u201d \u2014 quick... \ufffd"
    clean = normalize_text(dirty)
    assert "\u2019" not in clean
    assert "\u201c" not in clean
    assert "\u201d" not in clean
    assert "\ufffd" not in clean
    assert "What's the best way \"Fix Drain\" - quick... '" == clean


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for VRAM test")
def test_300m_vram_safety_ceiling():
    """Verify Robbie-300M forward/backward pass stays strictly below 5.0 GB VRAM."""
    config = TransformerConfig(
        vocab_size=2048,
        block_size=512,
        n_layer=24,
        n_head=16,
        n_embd=1024,
        dropout=0.0,
    )
    model = RobbieTransformer(config).cuda()
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
    except Exception:
        pytest.skip("bitsandbytes not installed; skipping 8-bit AdamW 300M VRAM step test.")

    guard = SafeVRAMGuard(limit_gb=4.8)

    x = torch.randint(0, 2048, (2, 512), device="cuda")
    y = torch.randint(0, 2048, (2, 512), device="cuda")
    scaler = torch.amp.GradScaler("cuda")

    with torch.autocast("cuda", dtype=torch.float16):
        _, loss = model(x, targets=y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

    del model, optimizer, x, y, loss
    torch.cuda.empty_cache()

    assert allocated_gb < 4.8, f"Allocated VRAM {allocated_gb:.2f} GB exceeded 4.8 GB ceiling!"
    assert reserved_gb < 5.0, f"Reserved VRAM {reserved_gb:.2f} GB exceeded 5.0 GB limit!"
