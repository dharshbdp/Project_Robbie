"""Character-level tokenizer implementation for Project Robbie."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from robbie.core.tokenizer.base import BaseTokenizer, SpecialTokens


class CharTokenizer(BaseTokenizer):
    """A deterministic character-level tokenizer.

    Each unique character maps to a unique token ID.
    Special tokens (<pad>, <unk>, <bos>, <eos>) are reserved at indices 0-3.
    """

    def __init__(self, special_tokens: Optional[SpecialTokens] = None):
        super().__init__(special_tokens=special_tokens)
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        """Assign fixed IDs (0-3) to special tokens."""
        self.token_to_id = {}
        self.id_to_token = {}
        for idx, token in enumerate(self.special_tokens.all_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def train(self, texts: List[str]) -> None:
        """Extract all unique characters from texts and build the vocabulary."""
        self._init_special_tokens()
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)

        # Sort characters for deterministic vocabulary ordering
        start_id = len(self.token_to_id)
        for idx, char in enumerate(sorted(unique_chars), start=start_id):
            self.token_to_id[char] = idx
            self.id_to_token[idx] = char

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Convert a string into a list of character token IDs."""
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.bos_token_id)

        for char in text:
            token_id = self.token_to_id.get(char, self.unk_token_id)
            ids.append(token_id)

        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Convert a list of token IDs back into a text string."""
        chars: List[str] = []
        special_set = set(self.special_tokens.all_tokens)

        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.special_tokens.UNK)
            if skip_special_tokens and token in special_set:
                continue
            chars.append(token)

        return "".join(chars)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save the tokenizer vocabulary to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "char",
            "special_tokens": {
                "pad": self.special_tokens.PAD,
                "unk": self.special_tokens.UNK,
                "bos": self.special_tokens.BOS,
                "eos": self.special_tokens.EOS,
            },
            "token_to_id": self.token_to_id,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CharTokenizer":
        """Load a saved CharTokenizer from a JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        st = SpecialTokens(
            PAD=data["special_tokens"]["pad"],
            UNK=data["special_tokens"]["unk"],
            BOS=data["special_tokens"]["bos"],
            EOS=data["special_tokens"]["eos"],
        )
        tokenizer = cls(special_tokens=st)
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        return tokenizer
