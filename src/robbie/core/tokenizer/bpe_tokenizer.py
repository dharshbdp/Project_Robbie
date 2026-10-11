"""Byte-Pair Encoding (BPE) subword tokenizer for Project Robbie."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from robbie.core.tokenizer.base import BaseTokenizer, SpecialTokens


class BPETokenizer(BaseTokenizer):
    """Byte-Pair Encoding (BPE) tokenizer implemented from scratch.

    Iteratively merges the most frequent adjacent character/subword pairs
    into new tokens until the desired vocabulary size is reached.
    """

    def __init__(
        self,
        target_vocab_size: int = 1000,
        special_tokens: Optional[SpecialTokens] = None,
    ):
        super().__init__(special_tokens=special_tokens)
        self.target_vocab_size = target_vocab_size
        self.merges: List[Tuple[str, str]] = []
        self._word_cache: Dict[str, List[str]] = {}
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        """Assign fixed IDs (0-3) to special tokens."""
        self.token_to_id = {}
        self.id_to_token = {}
        for idx, token in enumerate(self.special_tokens.all_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def _get_word_stats(self, texts: List[str]) -> Dict[Tuple[str, ...], int]:
        """Split text into words and count frequencies as tuples of characters."""
        word_freqs: Dict[Tuple[str, ...], int] = defaultdict(int)
        for text in texts:
            # Split by whitespace, preserving space boundaries by appending a boundary marker
            words = text.split()
            for word in words:
                if word:
                    # Represent word as tuple of individual characters
                    char_tuple = tuple(list(word))
                    word_freqs[char_tuple] += 1
        return word_freqs

    def _count_pairs(
        self, word_freqs: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, str], int]:
        """Count frequencies of adjacent symbol pairs across all words."""
        pairs: Dict[Tuple[str, str], int] = defaultdict(int)
        for word_tuple, freq in word_freqs.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                pairs[pair] += freq
        return pairs

    def _merge_word_tuple(
        self, word_tuple: Tuple[str, ...], pair: Tuple[str, str]
    ) -> Tuple[str, ...]:
        """Merge all occurrences of pair in a single word tuple."""
        p0, p1 = pair
        merged: List[str] = []
        i = 0
        while i < len(word_tuple):
            if (
                i < len(word_tuple) - 1
                and word_tuple[i] == p0
                and word_tuple[i + 1] == p1
            ):
                merged.append(p0 + p1)
                i += 2
            else:
                merged.append(word_tuple[i])
                i += 1
        return tuple(merged)

    def train(self, texts: List[str], min_freq: int = 2) -> None:
        """Train BPE merges and build vocabulary up to target_vocab_size."""
        self._init_special_tokens()
        self.merges = []
        self._word_cache.clear()

        # 1. Discover all base characters across the training corpus
        base_chars: Set[str] = set()
        for text in texts:
            base_chars.update(text)

        # Register base characters
        for char in sorted(base_chars):
            if char not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[char] = new_id
                self.id_to_token[new_id] = char

        # 2. Extract word frequencies
        word_freqs = self._get_word_stats(texts)
        if min_freq > 1:
            word_freqs = {w: f for w, f in word_freqs.items() if f >= min_freq}

        # 3. Iteratively merge most frequent pairs until target vocab size
        num_merges_to_perform = max(0, self.target_vocab_size - len(self.token_to_id))

        for _ in range(num_merges_to_perform):
            pairs = self._count_pairs(word_freqs)
            if not pairs:
                break

            # Find pair with highest frequency (break ties deterministically)
            best_pair = max(pairs.items(), key=lambda item: (item[1], item[0]))[0]
            if pairs[best_pair] < 1:
                break

            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)

            # Add new merged token to vocabulary
            new_id = len(self.token_to_id)
            self.token_to_id[new_token] = new_id
            self.id_to_token[new_id] = new_token

            # Update word frequencies: only re-merge words containing both pair elements
            p0, p1 = best_pair
            new_word_freqs: Dict[Tuple[str, ...], int] = {}
            for word_tuple, freq in word_freqs.items():
                if p0 in word_tuple and p1 in word_tuple:
                    merged_tuple = self._merge_word_tuple(word_tuple, best_pair)
                    new_word_freqs[merged_tuple] = freq
                else:
                    new_word_freqs[word_tuple] = freq
            word_freqs = new_word_freqs

    def _tokenize_word(self, word: str) -> List[str]:
        """Apply learned BPE merges to an individual word."""
        if not word:
            return []
        if word in self._word_cache:
            return self._word_cache[word]

        # Start with individual characters
        tokens = list(word)

        # Apply each learned merge in priority order
        for p0, p1 in self.merges:
            if p0 not in word or p1 not in word:
                continue
            i = 0
            merged: List[str] = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    merged.append(p0 + p1)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged

        self._word_cache[word] = tokens
        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text into token IDs using learned BPE merges."""
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.bos_token_id)

        # Split text while preserving spaces as separate tokens if needed
        # We handle whitespace by tokenizing words and inserting space tokens
        words = re.findall(r"\S+|\s+", text)
        for part in words:
            if part.isspace():
                # Encode whitespace characters directly
                for char in part:
                    token_id = self.token_to_id.get(char, self.unk_token_id)
                    ids.append(token_id)
            else:
                subwords = self._tokenize_word(part)
                for subword in subwords:
                    token_id = self.token_to_id.get(subword, self.unk_token_id)
                    ids.append(token_id)

        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back into string."""
        tokens: List[str] = []
        special_set = set(self.special_tokens.all_tokens)

        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.special_tokens.UNK)
            if skip_special_tokens and token in special_set:
                continue
            tokens.append(token)

        return "".join(tokens)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save BPE vocabulary and merge rules to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "bpe",
            "target_vocab_size": self.target_vocab_size,
            "special_tokens": {
                "pad": self.special_tokens.PAD,
                "unk": self.special_tokens.UNK,
                "bos": self.special_tokens.BOS,
                "eos": self.special_tokens.EOS,
            },
            "merges": self.merges,
            "token_to_id": self.token_to_id,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BPETokenizer":
        """Load a saved BPETokenizer from JSON."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        st = SpecialTokens(
            PAD=data["special_tokens"]["pad"],
            UNK=data["special_tokens"]["unk"],
            BOS=data["special_tokens"]["bos"],
            EOS=data["special_tokens"]["eos"],
        )
        tokenizer = cls(
            target_vocab_size=data.get("target_vocab_size", len(data["token_to_id"])),
            special_tokens=st,
        )
        tokenizer.merges = [tuple(m) for m in data.get("merges", [])]
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        return tokenizer
