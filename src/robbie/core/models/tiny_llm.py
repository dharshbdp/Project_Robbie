"""Tiny Neural Language Model for Project Robbie (Phase 2)."""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyLLMConfig:
    """Hyperparameters for TinyLLM."""
    vocab_size: int = 128
    block_size: int = 64       # Context window length
    n_embd: int = 128          # Embedding dimension
    n_hidden: int = 256        # Hidden feedforward dimension
    dropout: float = 0.1


class TinyLLM(nn.Module):
    """A small neural language model for next-token prediction.

    Architecture:
    Token Embeddings + Positional Embeddings
        ↓
    Feed-Forward Block (GELU + Dropout)
        ↓
    Layer Normalization + Residual Connection
        ↓
    Language Model Output Head (Vocab Projection)
    """

    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.fc1 = nn.Linear(config.n_embd, config.n_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.n_hidden, config.n_embd)
        self.ln = nn.LayerNorm(config.n_embd)

        # Output projection head: maps hidden state to unnormalized vocabulary logits
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between input embeddings and output head
        self.token_embedding.weight = self.head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            idx: (B, T) tensor of token integers
            targets: (B, T) tensor of target next-token integers

        Returns:
            logits: (B, T, vocab_size) unnormalized predictions
            loss: scalar cross-entropy loss if targets provided, else None
        """
        device = idx.device
        b, t = idx.shape

        if t > self.config.block_size:
            raise ValueError(
                f"Input sequence length ({t}) exceeds model block size ({self.config.block_size})"
            )

        # Position indices: [0, 1, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Token + Position Embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.pos_embedding(pos)    # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # Hidden layer with residual connection
        h = self.act(self.fc1(x))
        h = self.drop(self.fc2(h))
        x = self.ln(x + h)

        # Output logits
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten to compute Cross-Entropy Loss: (B * T, vocab_size) vs (B * T)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively generate next tokens given a conditioning prompt."""
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to context window if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass to get logits for the last token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            if temperature <= 0.0:
                # Greedy decoding (argmax)
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Scale logits by temperature
                logits = logits / temperature

                # Optional top-k filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled token to running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
