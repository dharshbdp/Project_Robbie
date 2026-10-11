"""Dataset loading and text preprocessing for Project Robbie."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset

from robbie.core.tokenizer.base import BaseTokenizer


class TextDataset(Dataset):
    """A PyTorch Dataset that converts tokenized text into input and target sequences.

    For next-token prediction:
        Input  x: tokens[i : i + seq_len]
        Target y: tokens[i + 1 : i + seq_len + 1]
    """

    def __init__(
        self,
        token_ids: List[int],
        seq_len: int = 64,
        stride: Optional[int] = None,
        device: str = "cpu",
    ):
        self.seq_len = seq_len
        self.stride = stride if stride is not None else 1
        self.device = device
        self.data = torch.tensor(token_ids, dtype=torch.long)

        if len(self.data) <= self.seq_len:
            raise ValueError(
                f"Data length ({len(self.data)}) must be greater than seq_len ({self.seq_len})"
            )

    def __len__(self) -> int:
        if self.stride == 1:
            return len(self.data) - self.seq_len
        return (len(self.data) - self.seq_len - 1) // self.stride + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x.to(self.device), y.to(self.device)


def load_text_file(filepath: Union[str, Path], encoding: str = "utf-8") -> str:
    """Load text content from a file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def split_text_data(
    text: str,
    train_ratio: float = 0.9,
) -> Tuple[str, str]:
    """Split text into training and validation sets."""
    split_idx = int(len(text) * train_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    return train_text, val_text


def prepare_datasets(
    text: str,
    tokenizer: BaseTokenizer,
    seq_len: int = 64,
    stride: Optional[int] = None,
    train_ratio: float = 0.9,
    device: str = "cpu",
) -> Tuple[TextDataset, TextDataset]:
    """Split text, tokenize, and return train and validation PyTorch datasets."""
    train_text, val_text = split_text_data(text, train_ratio=train_ratio)
    train_tokens = tokenizer.encode(train_text)
    val_tokens = tokenizer.encode(val_text)

    train_ds = TextDataset(train_tokens, seq_len=seq_len, stride=stride, device=device)
    val_ds = TextDataset(val_tokens, seq_len=seq_len, stride=stride, device=device)
    return train_ds, val_ds


def prepare_datasets_from_splits(
    train_text: str,
    val_text: str,
    tokenizer: BaseTokenizer,
    seq_len: int = 64,
    stride: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[TextDataset, TextDataset]:
    """Tokenize pre-split text strings and return train and validation PyTorch datasets.

    Defaults to non-overlapping chunks (stride = seq_len) to cover all tokens
    without redundant sequence duplication.
    """
    effective_stride = stride if stride is not None else seq_len
    train_tokens = tokenizer.encode(train_text)
    val_tokens = tokenizer.encode(val_text)
    train_ds = TextDataset(train_tokens, seq_len=seq_len, stride=effective_stride, device=device)
    val_ds = TextDataset(val_tokens, seq_len=seq_len, stride=effective_stride, device=device)
    return train_ds, val_ds

