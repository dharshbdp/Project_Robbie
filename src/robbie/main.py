"""Main CLI entrypoint for Project Robbie."""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List
import torch
from rich.console import Console
from rich.table import Table

from robbie import __version__
from robbie.config import RobbieConfig, DATA_DIR, CHECKPOINT_DIR
from robbie.utils.logger import logger
from robbie.utils.device import get_hardware_info, get_best_device
from robbie.core.tokenizer import CharTokenizer, BPETokenizer, prepare_datasets, load_text_file
from robbie.core.models.tiny_llm import TinyLLM, TinyLLMConfig
from robbie.core.models.transformer import RobbieTransformer, TransformerConfig
from robbie.core.models.inspector import (
    print_architecture_inspection,
    inspect_dataset_alignment,
    benchmark_generation,
)
from robbie.core.training.trainer import Trainer, TrainingConfig

BENCHMARK_PROMPTS = [
    "ROBBIE: ",
    "To be or not to be",
    "The king shall ",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Project Robbie CLI")
    parser.add_argument("--version", action="version", version=f"Project Robbie v{__version__}")
    parser.add_argument("--info", action="store_true", help="Print system and hardware report")
    parser.add_argument(
        "--demo-tokenizer",
        action="store_true",
        help="Demonstrate and compare CharTokenizer vs BPETokenizer",
    )
    parser.add_argument(
        "--train-tiny",
        action="store_true",
        help="Train the Phase 2 Tiny Language Model on the starter dataset",
    )
    parser.add_argument(
        "--train-transformer",
        action="store_true",
        help="Train the Phase 3 Causal Transformer (GPT) on the starter dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs for training (default: 5)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (number of non-improving epochs before stopping)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability for regularization (default: 0.1)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay for regularization (default: 0.01)",
    )
    parser.add_argument(
        "--inspect-baseline",
        action="store_true",
        help="Run comprehensive inspection of model architecture, tokenizer, loss, and benchmark samples",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run standardized benchmark prompts on current checkpoint",
    )
    parser.add_argument(
        "--compare-checkpoints",
        action="store_true",
        help="Run comparative benchmark between Epoch 3 (best val loss) and Epoch 5",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint file to load (default: best_transformer.pt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic generation",
    )
    parser.add_argument(
        "--generate",
        type=str,
        default=None,
        help="Generate text from the trained model given a starting prompt",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=["auto", "transformer", "tiny"],
        help="Which model checkpoint to use for generation (default: auto)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    return parser.parse_args()

def run_tokenizer_demo():
    print("\n" + "=" * 50)
    print("      PROJECT ROBBIE — TOKENIZER DEMONSTRATION")
    print("=" * 50)

    sample_training_corpus = [
        "Robbie is a personal AI character built from the ground up.",
        "Robbie has conversational memory, web research, and tool awareness.",
        "The language model learns to predict the next token in sequence.",
        "Deep learning transformers process attention across tokens.",
    ]
    test_phrase = "Robbie is learning language models and deep attention!"

    print(f"\n[Test Input String]:\n\"{test_phrase}\"\n")

    char_tok = CharTokenizer()
    char_tok.train(sample_training_corpus)
    char_encoded = char_tok.encode(test_phrase, add_special_tokens=True)
    char_decoded = char_tok.decode(char_encoded, skip_special_tokens=True)

    print("-" * 50)
    print("1. Character-Level Tokenizer (CharTokenizer)")
    print(f"   Vocabulary Size:    {char_tok.vocab_size} tokens")
    print(f"   Encoded Sequence:   {len(char_encoded)} token IDs")
    print(f"   Token IDs (first 15): {char_encoded[:15]}...")
    print(f"   Decoded Text:       \"{char_decoded}\"")

    bpe_tok = BPETokenizer(target_vocab_size=120)
    bpe_tok.train(sample_training_corpus)
    bpe_encoded = bpe_tok.encode(test_phrase, add_special_tokens=True)
    bpe_decoded = bpe_tok.decode(bpe_encoded, skip_special_tokens=True)

    print("\n2. Byte-Pair Encoding Subword Tokenizer (BPETokenizer)")
    print(f"   Learned Merges:     {len(bpe_tok.merges)} merges")
    print(f"   Vocabulary Size:    {bpe_tok.vocab_size} tokens")
    print(f"   Encoded Sequence:   {len(bpe_encoded)} token IDs")
    print(f"   Token IDs:          {bpe_encoded}")
    print(f"   Decoded Text:       \"{bpe_decoded}\"")

    compression = (1 - (len(bpe_encoded) / len(char_encoded))) * 100
    print(f"\n   Sequence Compression: {compression:.1f}% fewer tokens than character-level!")
    print("=" * 50 + "\n")

def run_train_tiny(epochs: int = 5):
    """Train TinyLLM on starter dataset."""
    dataset_file = DATA_DIR / "raw" / "tiny_shakespeare.txt"
    if not dataset_file.exists():
        logger.info(f"Dataset not found at {dataset_file}. Running downloader...")
        from scripts.download_data import download_dataset
        download_dataset()

    text = load_text_file(dataset_file)
    logger.info(f"Loaded dataset: {len(text):,} characters")

    tokenizer = CharTokenizer()
    tokenizer.train([text])
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(CHECKPOINT_DIR / "tiny_vocab.json")
    logger.info(f"Tokenizer trained with vocab size: {tokenizer.vocab_size}")

    block_size = 64
    train_ds, val_ds = prepare_datasets(text, tokenizer, seq_len=block_size, train_ratio=0.9)
    logger.info(f"Dataset split: {len(train_ds):,} training sequences, {len(val_ds):,} validation sequences")

    model_config = TinyLLMConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_embd=128,
        n_hidden=256,
        dropout=0.1,
    )
    model = TinyLLM(model_config)

    train_config = TrainingConfig(
        epochs=epochs,
        batch_size=64,
        learning_rate=1e-3,
        checkpoint_dir=str(CHECKPOINT_DIR),
        device="auto",
    )
    trainer = Trainer(model, train_ds, val_ds, train_config)
    trainer.fit()

    print("\n" + "=" * 50)
    print("      TRAINING COMPLETE! BEST MODEL SAVED TO:")
    print(f"      {CHECKPOINT_DIR / 'best_model.pt'}")
    print("=" * 50 + "\n")

def run_train_transformer(
    epochs: int = 5,
    patience: Optional[int] = None,
    dropout: float = 0.1,
    weight_decay: float = 0.01,
    learning_rate: float = 5e-4,
):
    """Train RobbieTransformer (Phase 3 GPT) on starter dataset."""
    dataset_file = DATA_DIR / "raw" / "tiny_shakespeare.txt"
    if not dataset_file.exists():
        logger.info(f"Dataset not found at {dataset_file}. Running downloader...")
        from scripts.download_data import download_dataset
        download_dataset()

    text = load_text_file(dataset_file)
    logger.info(f"Loaded dataset: {len(text):,} characters")

    # Use or train tokenizer
    vocab_file = CHECKPOINT_DIR / "transformer_vocab.json"
    tokenizer = CharTokenizer()
    tokenizer.train([text])
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(vocab_file)
    logger.info(f"Tokenizer ready with vocab size: {tokenizer.vocab_size}")

    block_size = 64
    train_ds, val_ds = prepare_datasets(text, tokenizer, seq_len=block_size, train_ratio=0.9)
    logger.info(f"Dataset split: {len(train_ds):,} training sequences, {len(val_ds):,} validation sequences")

    # Configure RobbieTransformer: 4 Transformer Blocks, 4 Attention Heads, 128 Embedding Dim
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=dropout,
    )
    model = RobbieTransformer(config)

    train_config = TrainingConfig(
        epochs=epochs,
        batch_size=64,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        checkpoint_dir=str(CHECKPOINT_DIR),
        best_model_name="best_transformer.pt",
        early_stopping_patience=patience,
        device="auto",
    )
    trainer = Trainer(model, train_ds, val_ds, train_config)
    history = trainer.fit()

    best_transformer_path = CHECKPOINT_DIR / "best_transformer.pt"
    best_loss = history.get("best_val_loss", "N/A")
    loss_display = f"{best_loss:.4f}" if isinstance(best_loss, float) else str(best_loss)
    print("\n" + "=" * 55)
    print("   TRANSFORMER TRAINING COMPLETE! BEST MODEL SAVED TO:")
    print(f"   {best_transformer_path}")
    print(f"   Best Epoch: {history.get('best_epoch', 'N/A')} | Lowest Val Loss: {loss_display}")
    print("=" * 55 + "\n")

def run_inspect_baseline():
    """Execute all 4 baseline verification checks."""
    console = Console()
    console.print("\n" + "=" * 65)
    console.print("   ROBBIE v0.2 BASELINE VERIFICATION & QUALITY AUDIT")
    console.print("=" * 65 + "\n")

    vocab_file = CHECKPOINT_DIR / "tiny_vocab.json"
    checkpoint_file = CHECKPOINT_DIR / "best_model.pt"
    dataset_file = DATA_DIR / "raw" / "tiny_shakespeare.txt"
    history_file = CHECKPOINT_DIR / "training_history.json"

    if not vocab_file.exists():
        console.print("[red]Vocabulary file not found. Train first with: python -m robbie.main --train-tiny[/]")
        return

    tokenizer = CharTokenizer.load(vocab_file)
    model_config = TinyLLMConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=64,
        n_embd=128,
        n_hidden=256,
    )
    model = TinyLLM(model_config)

    # 1. Architecture Inspection
    console.print("[bold]1. MODEL ARCHITECTURE & PARAMETER AUDIT[/]")
    print_architecture_inspection(model)

    # 2. Tokenizer & Dataset Alignment
    if dataset_file.exists():
        console.print("[bold]2. TOKENIZER & DATASET ALIGNMENT VERIFICATION[/]")
        sample_text = load_text_file(dataset_file)
        inspect_dataset_alignment(sample_text, tokenizer, seq_len=12)

    # 3. Loss Tracking & Overfitting Analysis
    console.print("[bold]3. LOSS TRACKING & CONVERGENCE AUDIT[/]")
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)

        loss_table = Table(title="Training vs Validation Loss History")
        loss_table.add_column("Epoch", justify="right", style="cyan")
        loss_table.add_column("Train Loss", justify="right", style="green")
        loss_table.add_column("Val Loss", justify="right", style="yellow")
        loss_table.add_column("Gap (Val - Train)", justify="right", style="magenta")

        train_losses = history.get("train_loss", [])
        val_losses = history.get("val_loss", [])

        for i in range(len(train_losses)):
            t_l = train_losses[i]
            v_l = val_losses[i]
            gap = v_l - t_l
            loss_table.add_row(str(i + 1), f"{t_l:.4f}", f"{v_l:.4f}", f"{gap:+.4f}")

        console.print(loss_table)
        final_gap = val_losses[-1] - train_losses[-1]
        if final_gap < 0.1:
            console.print("[green]✓ Healthy Generalization:[/] Validation loss closely tracks training loss without overfitting divergence.\n")
        else:
            console.print("[yellow]⚠ Note:[/] Validation loss is higher than training loss by >0.1.\n")
    else:
        console.print("[yellow]No training_history.json found yet.[/]\n")

    # 4. Standardized Benchmark Generation
    console.print("[bold]4. STANDARDIZED BENCHMARK GENERATION[/]")
    if checkpoint_file.exists():
        Trainer.load_checkpoint(checkpoint_file, model)
        benchmark_generation(
            model=model,
            tokenizer=tokenizer,
            prompts=BENCHMARK_PROMPTS,
            max_tokens=60,
            temperature=0.8,
            seed=42,
            device=get_best_device(),
        )
    else:
        console.print("[yellow]No checkpoint found to generate benchmarks from.[/]")

def run_generate(
    prompt: str,
    model_type: str = "auto",
    checkpoint: Optional[str] = None,
    max_tokens: int = 100,
    temperature: float = 0.8,
    seed: Optional[int] = None,
):
    """Generate text from trained checkpoint."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if checkpoint is not None:
        selected_ckpt = Path(checkpoint)
        if not selected_ckpt.exists():
            print(f"[Error] Specified checkpoint not found: {selected_ckpt}")
            return
        # Infer architecture from state_dict keys
        raw_data = torch.load(selected_ckpt, map_location="cpu", weights_only=False)
        state_keys = list(raw_data.get("model_state_dict", {}).keys())
        is_transformer = any("blocks" in k for k in state_keys)
        model_type = "transformer" if is_transformer else "tiny"
    else:
        transformer_ckpt = CHECKPOINT_DIR / "best_transformer.pt"
        tiny_ckpt = CHECKPOINT_DIR / "best_model.pt"

        # Auto select model type
        if model_type == "auto":
            if transformer_ckpt.exists():
                selected_ckpt = transformer_ckpt
                model_type = "transformer"
            elif tiny_ckpt.exists():
                selected_ckpt = tiny_ckpt
                model_type = "tiny"
            else:
                print("[Error] No trained checkpoint found. Train first with: python -m robbie.main --train-transformer")
                return
        elif model_type == "transformer":
            selected_ckpt = transformer_ckpt
        else:
            selected_ckpt = tiny_ckpt

    device = get_best_device()

    if model_type == "transformer":
        vocab_file = CHECKPOINT_DIR / "transformer_vocab.json"
        if not vocab_file.exists():
            vocab_file = CHECKPOINT_DIR / "tiny_vocab.json"
        tokenizer = CharTokenizer.load(vocab_file)
        config = TransformerConfig(vocab_size=tokenizer.vocab_size, block_size=64, n_layer=4, n_head=4, n_embd=128)
        model = RobbieTransformer(config)
        Trainer.load_checkpoint(selected_ckpt, model)
        logger.info(f"Loaded RobbieTransformer checkpoint: {selected_ckpt.name}")
    else:
        vocab_file = CHECKPOINT_DIR / "tiny_vocab.json"
        tokenizer = CharTokenizer.load(vocab_file)
        config = TinyLLMConfig(vocab_size=tokenizer.vocab_size, block_size=64, n_embd=128, n_hidden=256)
        model = TinyLLM(config)
        Trainer.load_checkpoint(selected_ckpt, model)
        logger.info(f"Loaded TinyLLM checkpoint: {selected_ckpt.name}")

    model.to(device)
    model.eval()

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_token_id]
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    seed_str = str(seed) if seed is not None else "random"
    print(f"\n[Checkpoint]: {selected_ckpt.name} | [Seed]: {seed_str}")
    print(f"[Prompt]: \"{prompt}\"")
    print(f"[Generating with temperature {temperature}...]\n")
    generated_tensor = model.generate(input_tensor, max_new_tokens=max_tokens, temperature=temperature, top_k=20)
    generated_ids = generated_tensor[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("-" * 50)
    print(generated_text)
    print("-" * 50 + "\n")

def main():
    args = parse_args()
    logger.info(f"Starting Project Robbie (v{__version__})")

    if args.info:
        report = get_hardware_info()
        print("\n=== Project Robbie Hardware Report ===")
        print(f"Device Type:     {report.device_type}")
        print(f"Device Name:     {report.device_name}")
        print(f"CUDA Available:  {report.cuda_available}")
        print(f"GPU Count:       {report.device_count}")
        if report.cuda_available:
            print(f"Total VRAM:      {report.vram_total_gb} GB")
            print(f"Available VRAM:  {report.vram_free_gb} GB")
        print("======================================\n")

    if args.demo_tokenizer:
        run_tokenizer_demo()

    if args.train_tiny:
        run_train_tiny(epochs=args.epochs)

    if args.train_transformer:
        run_train_transformer(
            epochs=args.epochs,
            patience=args.patience,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
        )

    if args.compare_checkpoints:
        from scripts.compare_checkpoints import compare_checkpoints
        ckpt_a = Path("checkpoints/transformer_epoch3_best.pt")
        ckpt_b = Path("checkpoints/transformer_epoch5.pt")
        if ckpt_a.exists() and ckpt_b.exists():
            compare_checkpoints(
                ckpt_a=ckpt_a,
                ckpt_b=ckpt_b,
                label_a="Epoch 3 (Best Val Loss: 1.5024)",
                label_b="Epoch 5 (Final: 1.5056)",
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=args.seed if args.seed is not None else 42,
            )
        else:
            print("[Error] One or both checkpoints (epoch 3 / epoch 5) not found.")

    if args.inspect_baseline:
        run_inspect_baseline()

    if args.benchmark:
        vocab_file = CHECKPOINT_DIR / "tiny_vocab.json"
        checkpoint_file = CHECKPOINT_DIR / "best_model.pt"
        if vocab_file.exists() and checkpoint_file.exists():
            tokenizer = CharTokenizer.load(vocab_file)
            model_config = TinyLLMConfig(vocab_size=tokenizer.vocab_size, block_size=64, n_embd=128, n_hidden=256)
            model = TinyLLM(model_config)
            Trainer.load_checkpoint(checkpoint_file, model)
            benchmark_generation(
                model=model,
                tokenizer=tokenizer,
                prompts=BENCHMARK_PROMPTS,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=42,
                device=get_best_device(),
            )
        else:
            print("[Error] No trained checkpoint found. Train first with: python -m robbie.main --train-tiny")

    if args.generate is not None:
        run_generate(
            args.generate,
            model_type=args.model,
            checkpoint=args.checkpoint,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )

if __name__ == "__main__":
    main()
