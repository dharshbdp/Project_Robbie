"""RobbieTransformer architecture for Project Robbie (Phase 3)."""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from robbie.core.models.transformer_block import TransformerBlock


@dataclass
class TransformerConfig:
    """Hyperparameters for RobbieTransformer."""
    vocab_size: int = 128
    block_size: int = 64     # Maximum context window length
    n_layer: int = 4         # Number of stacked Transformer blocks
    n_head: int = 4          # Number of attention heads
    n_embd: int = 128        # Hidden embedding dimension
    dropout: float = 0.1     # Dropout probability


class RobbieTransformer(nn.Module):
    """Generative Pretrained Transformer (GPT) language model.

    Combines token & positional embeddings, multiple layers of Causal
    Multi-Head Attention and MLPs, with Pre-LayerNorm and weight tying.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token and Positional Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embd=config.n_embd,
                n_head=config.n_head,
                block_size=config.block_size,
                dropout=config.dropout,
            )
            for _ in range(config.n_layer)
        ])

        # Final LayerNorm before language modeling head
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Language Model Head (projects hidden states to vocabulary logits)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between input token embedding and output head
        self.wte.weight = self.lm_head.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            idx: (Batch, Seq_Len) tensor of token indices
            targets: Optional (Batch, Seq_Len) tensor of target next-token indices

        Returns:
            logits: (Batch, Seq_Len, Vocab_Size)
            loss: Cross-entropy loss scalar if targets provided, else None
        """
        device = idx.device
        b, t = idx.shape

        if t > self.config.block_size:
            raise ValueError(
                f"Sequence length ({t}) exceeds model block size ({self.config.block_size})"
            )

        # Positions: [0, 1, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Compute embeddings
        tok_emb = self.wte(idx)  # (B, T, n_embd)
        pos_emb = self.wpe(pos)  # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # Pass through stacked Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply final LayerNorm
        x = self.ln_f(x)

        # Project to vocabulary logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 60,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
    ) -> torch.Tensor:
        """Autoregressively sample tokens conditioned on input prompt."""
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to context window if length exceeds block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Logits for next token (B, vocab_size)

            if temperature <= 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
