"""Causal Multi-Head Self-Attention implementation for Project Robbie (Phase 3)."""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention mechanism.

    Allows tokens in a sequence to attend to all preceding tokens,
    capturing long-range contextual and grammatical dependencies.
    """

    def __init__(
        self,
        n_embd: int = 128,
        n_head: int = 4,
        block_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size

        # Key, Query, Value projections combined into a single linear layer
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)

        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower-triangular matrix ensuring tokens only attend to past positions
        # Shape: (1, 1, block_size, block_size)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for causal multi-head attention.

        Args:
            x: Input tensor of shape (Batch, Seq_Len, Embedding_Dim)

        Returns:
            Output tensor of shape (Batch, Seq_Len, Embedding_Dim)
        """
        B, T, C = x.shape  # Batch size, Sequence length, Embedding dimension (n_embd)

        # 1. Project Q, K, V in a single pass: (B, T, 3 * C)
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape into heads: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Scaled dot-product attention scores: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 4. Apply causal mask: mask future positions with -inf
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # 5. Softmax to get attention probabilities
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 6. Weight values by attention probabilities: (B, n_head, T, head_dim)
        y = att @ v

        # 7. Concatenate all heads back together: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Output projection with residual dropout
        return self.resid_dropout(self.c_proj(y))
