"""Transformer Block and Feed-Forward MLP for Project Robbie (Phase 3)."""

import torch
import torch.nn as nn
from robbie.core.models.attention import CausalSelfAttention


class MLP(nn.Module):
    """Feed-Forward Network (MLP) within each Transformer block."""

    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        # Standard 4x expansion factor
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """A single Transformer Decoder layer with Pre-LayerNorm architecture.

    Structure:
    Input x ──► LayerNorm ──► Causal Multi-Head Attention ──► (+) Residual
              │                                                │
              └──► LayerNorm ──► Feed-Forward MLP ─────────────► (+) Residual
    """

    def __init__(
        self,
        n_embd: int = 128,
        n_head: int = 4,
        block_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
        )
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
