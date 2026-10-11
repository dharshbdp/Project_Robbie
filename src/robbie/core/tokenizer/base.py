"""Base tokenizer interface and special tokens for Project Robbie."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

@dataclass(frozen=True)
class SpecialTokens:
    """Standard special tokens used across Robbie tokenizers."""
    PAD: str = "<pad>"  # Padding token
    UNK: str = "<unk>"  # Unknown token
    BOS: str = "<bos>"  # Beginning of sequence
    EOS: str = "<eos>"  # End of sequence

    @property
    def all_tokens(self) -> List[str]:
        return [self.PAD, self.UNK, self.BOS, self.EOS]


class BaseTokenizer(ABC):
    """Abstract base class for all Robbie tokenizers."""

    def __init__(self, special_tokens: Optional[SpecialTokens] = None):
        self.special_tokens = special_tokens or SpecialTokens()
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        """Total number of unique tokens in the vocabulary."""
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.special_tokens.PAD]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.special_tokens.UNK]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.special_tokens.BOS]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.special_tokens.EOS]

    @abstractmethod
    def train(self, texts: List[str]) -> None:
        """Train the tokenizer vocabulary on a collection of texts."""
        pass

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Convert a string into a list of token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Convert a list of token IDs back into a string."""
        pass

    @abstractmethod
    def save(self, filepath: Union[str, Path]) -> None:
        """Save the tokenizer vocabulary and state to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseTokenizer":
        """Load a saved tokenizer from disk."""
        pass
