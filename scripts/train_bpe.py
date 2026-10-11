"""Train and benchmark BPE subword tokenizer on Stage 1 Pretraining corpus."""

import argparse
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table

from robbie.core.tokenizer.bpe_tokenizer import BPETokenizer
from robbie.core.tokenizer.char_tokenizer import CharTokenizer
from robbie.config import DATA_DIR, CHECKPOINT_DIR
from robbie.utils.logger import logger


def train_stage1_bpe(
    vocab_size: int = 1024,
    min_freq: int = 3,
    sample_chars: int = 2_000_000,
    output_path: Path = CHECKPOINT_DIR / "bpe_stage1_vocab.json",
) -> BPETokenizer:
    """Train BPE tokenizer on Stage 1 pretraining text."""
    console = Console()
    console.print("\n" + "=" * 65)
    console.print(f"   PROJECT ROBBIE — STAGE 1 BPE TOKENIZER TRAINING")
    console.print(f"   Target Vocab Size: {vocab_size} | Min Frequency: {min_freq}")
    console.print("=" * 65 + "\n")

    train_file = DATA_DIR / "processed" / "stage1" / "train.txt"
    if not train_file.exists():
        console.print(f"[red]Stage 1 train file not found at {train_file}.[/]")
        console.print("Run: python scripts/prepare_data.py --stage 1 first.")
        raise FileNotFoundError(f"Missing {train_file}")

    console.print(f"[cyan]Loading Stage 1 training text from {train_file.name}...[/]")
    with open(train_file, "r", encoding="utf-8") as f:
        # Read training corpus (use sample for rapid training if corpus is huge)
        full_text = f.read()

    text_to_train = full_text[:sample_chars] if len(full_text) > sample_chars else full_text
    # Remove literal <unk> tags originating from WikiText so they don't form corrupted merges
    text_to_train = text_to_train.replace("<unk>", " ")
    console.print(f"[dim]Training corpus size: {len(text_to_train):,} characters ({len(text_to_train.split()):,} words)[/]")

    tokenizer = BPETokenizer(target_vocab_size=vocab_size)

    console.print("[yellow]Discovering base characters and learning merge rules...[/]")
    t0 = time.time()
    tokenizer.train([text_to_train], min_freq=min_freq)
    elapsed = time.time() - t0

    console.print(f"[green]Training completed in {elapsed:.2f} seconds![/]")
    console.print(f"Total Merges Learned: [bold]{len(tokenizer.merges)}[/]")
    console.print(f"Final Vocabulary Size: [bold]{tokenizer.vocab_size}[/] tokens\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)
    console.print(f"[green]Saved trained BPE tokenizer to: {output_path}[/]\n")

    # Benchmarking Compression against Character Tokenizer
    benchmark_phrases = [
        "Robbie is a personal artificial intelligence assistant built from scratch.",
        "The transformer architecture uses multi-head self-attention mechanisms.",
        "In Python, neural networks optimize loss using gradient descent.",
        "User: Hello Robbie, can you help me understand Byte-Pair Encoding?\nRobbie: Certainly! BPE merges frequent pairs.",
    ]

    table = Table(title="Compression Benchmark: BPE Subword vs Character-Level")
    table.add_column("Test Phrase", style="cyan", width=38)
    table.add_column("Char Tokens", style="yellow", justify="right")
    table.add_column("BPE Tokens", style="green", justify="right")
    table.add_column("Compression", style="magenta", justify="right")

    for phrase in benchmark_phrases:
        char_tokens = len(phrase)
        bpe_tokens = len(tokenizer.encode(phrase))
        compression = (1 - (bpe_tokens / char_tokens)) * 100
        short_phrase = (phrase[:35] + "...") if len(phrase) > 35 else phrase
        table.add_row(
            short_phrase,
            f"{char_tokens}",
            f"{bpe_tokens}",
            f"{compression:.1f}% fewer",
        )

    console.print(table)
    console.print()
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 BPE Tokenizer for Project Robbie.")
    parser.add_argument("--vocab-size", type=int, default=1024, help="Target vocabulary size (default: 1024)")
    parser.add_argument("--min-freq", type=int, default=3, help="Minimum word frequency to consider for merges (default: 3)")
    parser.add_argument("--sample-chars", type=int, default=2_500_000, help="Number of characters to train BPE merges on (default: 2.5M)")
    args = parser.parse_args()

    train_stage1_bpe(
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        sample_chars=args.sample_chars,
    )


if __name__ == "__main__":
    main()
