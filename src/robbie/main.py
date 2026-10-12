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
from robbie.core.tokenizer import (
    CharTokenizer,
    BPETokenizer,
    prepare_datasets,
    prepare_datasets_from_splits,
    load_text_file,
)
from robbie.core.models.tiny_llm import TinyLLM, TinyLLMConfig
from robbie.core.models.transformer import RobbieTransformer, TransformerConfig
from robbie.core.models.inspector import (
    print_architecture_inspection,
    inspect_dataset_alignment,
    benchmark_generation,
)
from robbie.core.training.trainer import Trainer, TrainingConfig
from robbie.core.conversation import (
    Role,
    Message,
    ConversationFormatter,
    ConversationContext,
    ChatEngine,
    ConversationEvaluator,
)

BENCHMARK_PROMPTS = [
    "ROBBIE: ",
    "To be or not to be",
    "The king shall ",
]

MODEL_PRESETS = {
    "prototype": {
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "stage1_ckpt": "stage1_base_transformer.pt",
        "stage2_ckpt": "stage2_conversational_transformer.pt",
        "name": "Robbie-Prototype (~808k parameters)",
    },
    "11m": {
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "stage1_ckpt": "stage1_11m_base_transformer.pt",
        "stage2_ckpt": "stage2_11m_conversational_transformer.pt",
        "name": "Robbie-11M (~11.1 Million parameters)",
    },
    "25m": {
        "n_layer": 8,
        "n_head": 8,
        "n_embd": 512,
        "stage1_ckpt": "stage1_25m_base_transformer.pt",
        "stage2_ckpt": "stage2_25m_conversational_transformer.pt",
        "name": "Robbie-25M (~25.8 Million parameters)",
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Project Robbie CLI")
    parser.add_argument("--version", action="version", version=f"Project Robbie v{__version__}")
    parser.add_argument("--info", action="store_true", help="Print system and hardware report")
    parser.add_argument(
        "--model-size",
        type=str,
        default="25m",
        choices=["prototype", "11m", "25m"],
        help="Model architecture scale: 'prototype' (~808k), '11m' (~11M), or '25m' (~25.8M) (default: 25m)",
    )
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
        "--train-stage1",
        action="store_true",
        help="Train the Phase 4 Stage 1 Base Transformer with BPE subwords on WikiText",
    )
    parser.add_argument(
        "--train-stage2",
        action="store_true",
        help="Fine-tune the Phase 4 Stage 2 Conversational Transformer on assistant dialogues",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Launch an interactive terminal chat session with Robbie",
    )
    parser.add_argument(
        "--eval-conversation",
        action="store_true",
        help="Run Phase 5 Conversational Model quality & multi-turn evaluation benchmark",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Context window length in tokens (default: 128)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride step size between consecutive sequences (default: auto)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable Automatic Mixed Precision (AMP) on CUDA",
    )
    parser.add_argument(
        "--no-cosine",
        action="store_true",
        help="Disable Cosine Decay learning rate scheduler",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs for training (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
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
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.25,
        help="Penalty for previously generated tokens to prevent repetition loops (default: 1.25)",
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

def run_train_stage1(
    model_size: str = "25m",
    epochs: int = 5,
    batch_size: int = 32,
    block_size: int = 128,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.03,
    dropout: float = 0.1,
    patience: Optional[int] = 2,
    use_amp: bool = True,
    use_cosine: bool = True,
):
    """Train Robbie Stage 1 Base Transformer with BPE subwords on WikiText + TinyStories."""
    console = Console()
    preset = MODEL_PRESETS.get(model_size, MODEL_PRESETS["25m"])
    save_ckpt_name = preset["stage1_ckpt"]

    console.print("\n" + "=" * 65)
    console.print(f"   PROJECT ROBBIE — STAGE 1 BASE PRETRAINING: {preset['name']}")
    console.print(f"   Context Window: {block_size} tokens | Batch Size: {batch_size}")
    console.print("=" * 65 + "\n")

    train_file = DATA_DIR / "processed" / "stage1" / "train.txt"
    val_file = DATA_DIR / "processed" / "stage1" / "val.txt"
    bpe_vocab_file = CHECKPOINT_DIR / "bpe_stage1_vocab.json"

    # 1. Ensure dataset exists
    if not train_file.exists() or not val_file.exists():
        logger.info("Stage 1 dataset not found. Running dataset pipeline...")
        from robbie.core.data import DatasetPipeline
        pipeline = DatasetPipeline(data_root=DATA_DIR)
        pipeline.prepare_stage1_pretrain()

    # 2. Ensure BPE tokenizer exists
    if not bpe_vocab_file.exists():
        logger.info("Stage 1 BPE vocabulary not found. Training BPE tokenizer...")
        from scripts.train_bpe import train_stage1_bpe
        tokenizer = train_stage1_bpe(vocab_size=2048, output_path=bpe_vocab_file)
    else:
        tokenizer = BPETokenizer.load(bpe_vocab_file)
        logger.info(f"Loaded BPE tokenizer with {tokenizer.vocab_size} tokens from {bpe_vocab_file.name}")

    # 3. Load text and prepare datasets
    logger.info("Loading preprocessed Stage 1 text files...")
    train_text = load_text_file(train_file)
    val_text = load_text_file(val_file)

    logger.info(f"Tokenizing Stage 1 text (train: {len(train_text):,} chars, val: {len(val_text):,} chars)...")
    train_ds, val_ds = prepare_datasets_from_splits(
        train_text, val_text, tokenizer=tokenizer, seq_len=block_size
    )
    logger.info(f"Dataset split: {len(train_ds):,} training sequences, {len(val_ds):,} validation sequences")

    # 4. Configure Stage 1 Base Transformer
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=preset["n_layer"],
        n_head=preset["n_head"],
        n_embd=preset["n_embd"],
        dropout=dropout,
    )
    model = RobbieTransformer(config)

    train_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=1.0,
        use_amp=use_amp,
        use_cosine_lr=use_cosine,
        checkpoint_dir=str(CHECKPOINT_DIR),
        best_model_name=save_ckpt_name,
        early_stopping_patience=patience,
        device="auto",
    )
    trainer = Trainer(model, train_ds, val_ds, train_config)
    history = trainer.fit()

    best_path = CHECKPOINT_DIR / save_ckpt_name
    best_loss = history.get("best_val_loss", "N/A")
    loss_disp = f"{best_loss:.4f}" if isinstance(best_loss, float) else str(best_loss)
    best_ppl = history.get("best_val_perplexity", "N/A")
    ppl_disp = f"{best_ppl:.2f}" if isinstance(best_ppl, float) else str(best_ppl)

    console.print("\n" + "=" * 65)
    console.print(f"   STAGE 1 BASE PRETRAINING COMPLETE ({preset['name']})!")
    console.print(f"   Best Model Saved To: [bold green]{best_path}[/]")
    console.print(f"   Best Epoch: {history.get('best_epoch', 'N/A')} | Lowest Val Loss: {loss_disp} | Perplexity: {ppl_disp}")
    console.print("=" * 65 + "\n")

def run_train_stage2(
    model_size: str = "25m",
    epochs: int = 25,
    batch_size: int = 16,
    block_size: int = 128,
    stride: int = 16,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    dropout: float = 0.1,
    patience: Optional[int] = 5,
    use_amp: bool = True,
    use_cosine: bool = True,
):
    """Fine-tune Stage 1 Base Robbie Transformer on Stage 2 Assistant Dialogues."""
    console = Console()
    preset = MODEL_PRESETS.get(model_size, MODEL_PRESETS["25m"])
    stage1_base_path = CHECKPOINT_DIR / preset["stage1_ckpt"]
    save_ckpt_name = preset["stage2_ckpt"]

    console.print("\n" + "=" * 65)
    console.print(f"   PROJECT ROBBIE — STAGE 2 CONVERSATIONAL FINE-TUNING: {preset['name']}")
    console.print(f"   Context Window: {block_size} tokens | Batch Size: {batch_size} | Stride: {stride}")
    console.print("=" * 65 + "\n")

    train_file = DATA_DIR / "processed" / "stage2" / "train.txt"
    val_file = DATA_DIR / "processed" / "stage2" / "val.txt"
    bpe_vocab_file = CHECKPOINT_DIR / "bpe_stage1_vocab.json"

    # 1. Ensure Stage 1 base model exists
    if not stage1_base_path.exists():
        console.print(
            f"[red]Stage 1 Base checkpoint not found at {stage1_base_path}. "
            f"Please train Stage 1 base model first: python -m robbie.main --train-stage1 --model-size {model_size}[/]"
        )
        return

    # 2. Ensure dataset exists
    if not train_file.exists() or not val_file.exists():
        logger.info("Stage 2 dataset not found. Running dialogue dataset pipeline...")
        from robbie.core.data import DatasetPipeline
        pipeline = DatasetPipeline(data_root=DATA_DIR)
        pipeline.prepare_stage2_dialogue(force=True)

    # 3. Load BPE tokenizer
    if not bpe_vocab_file.exists():
        console.print(f"[red]BPE vocabulary file not found at {bpe_vocab_file}[/]")
        return
    tokenizer = BPETokenizer.load(bpe_vocab_file)
    logger.info(f"Loaded BPE tokenizer with {tokenizer.vocab_size} tokens from {bpe_vocab_file.name}")

    # 4. Load text and prepare datasets
    logger.info("Loading preprocessed Stage 2 dialogue text files...")
    train_text = load_text_file(train_file)
    val_text = load_text_file(val_file)

    logger.info(f"Tokenizing Stage 2 text (train: {len(train_text):,} chars, val: {len(val_text):,} chars)...")
    train_ds, val_ds = prepare_datasets_from_splits(
        train_text, val_text, tokenizer=tokenizer, seq_len=block_size, stride=stride
    )
    logger.info(f"Dataset split: {len(train_ds):,} training sequences, {len(val_ds):,} validation sequences")

    # 5. Configure Model and load Stage 1 Base weights
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=preset["n_layer"],
        n_head=preset["n_head"],
        n_embd=preset["n_embd"],
        dropout=dropout,
    )
    model = RobbieTransformer(config)
    Trainer.load_checkpoint(stage1_base_path, model)
    logger.info(f"Successfully loaded pretrained Stage 1 base weights from {stage1_base_path.name}")

    # 6. Fine-tune with low learning rate and cosine decay
    train_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=1.0,
        use_amp=use_amp,
        use_cosine_lr=use_cosine,
        checkpoint_dir=str(CHECKPOINT_DIR),
        best_model_name=save_ckpt_name,
        early_stopping_patience=patience,
        device="auto",
    )
    trainer = Trainer(model, train_ds, val_ds, train_config)
    history = trainer.fit()

    best_path = CHECKPOINT_DIR / save_ckpt_name
    latest_path = CHECKPOINT_DIR / "latest_model.pt"
    # Ensure the conversational checkpoint retains the fully aligned weights
    if latest_path.exists():
        import shutil
        shutil.copy2(latest_path, best_path)

    best_loss = history.get("best_val_loss", "N/A")
    loss_disp = f"{best_loss:.4f}" if isinstance(best_loss, float) else str(best_loss)
    best_ppl = history.get("best_val_perplexity", "N/A")
    ppl_disp = f"{best_ppl:.2f}" if isinstance(best_ppl, float) else str(best_ppl)

    console.print("\n" + "=" * 65)
    console.print(f"   STAGE 2 CONVERSATIONAL FINE-TUNING COMPLETE ({preset['name']})!")
    console.print(f"   Conversational Model Saved To: [bold green]{best_path}[/]")
    console.print(f"   Best Epoch: {history.get('best_epoch', 'N/A')} | Lowest Val Loss: {loss_disp} | Perplexity: {ppl_disp}")
    console.print("=" * 65 + "\n")

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
    temperature: float = 0.7,
    repetition_penalty: float = 1.25,
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
        stage2_25m_ckpt = CHECKPOINT_DIR / "stage2_25m_conversational_transformer.pt"
        stage1_25m_ckpt = CHECKPOINT_DIR / "stage1_25m_base_transformer.pt"
        stage2_11m_ckpt = CHECKPOINT_DIR / "stage2_11m_conversational_transformer.pt"
        stage1_11m_ckpt = CHECKPOINT_DIR / "stage1_11m_base_transformer.pt"
        stage2_ckpt = CHECKPOINT_DIR / "stage2_conversational_transformer.pt"
        stage1_ckpt = CHECKPOINT_DIR / "stage1_base_transformer.pt"
        transformer_ckpt = CHECKPOINT_DIR / "best_transformer.pt"
        tiny_ckpt = CHECKPOINT_DIR / "best_model.pt"

        available_transformers = [
            stage2_25m_ckpt,
            stage1_25m_ckpt,
            stage2_11m_ckpt,
            stage1_11m_ckpt,
            stage2_ckpt,
            stage1_ckpt,
            transformer_ckpt,
        ]

        # Auto select model type
        if model_type == "auto":
            found_tf = next((c for c in available_transformers if c.exists()), None)
            if found_tf is not None:
                selected_ckpt = found_tf
                model_type = "transformer"
            elif tiny_ckpt.exists():
                selected_ckpt = tiny_ckpt
                model_type = "tiny"
            else:
                print("[Error] No trained checkpoint found. Train first with: python -m robbie.main --train-stage1")
                return
        elif model_type == "transformer":
            found_tf = next((c for c in available_transformers if c.exists()), None)
            selected_ckpt = found_tf if found_tf is not None else transformer_ckpt
        else:
            selected_ckpt = tiny_ckpt

    raw_data = torch.load(selected_ckpt, map_location="cpu", weights_only=False)
    saved_config = raw_data.get("config", None)
    bpe_vocab_file = CHECKPOINT_DIR / "bpe_stage1_vocab.json"
    is_bpe = bpe_vocab_file.exists() and (
        "stage" in selected_ckpt.name
        or (saved_config is not None and getattr(saved_config, "vocab_size", 0) > 500)
    )

    device = get_best_device()

    if model_type == "transformer":
        if is_bpe:
            tokenizer = BPETokenizer.load(bpe_vocab_file)
            block_size = getattr(saved_config, "block_size", 128) if saved_config else 128
            if isinstance(saved_config, TransformerConfig):
                config = saved_config
            elif isinstance(saved_config, dict):
                config = TransformerConfig(**saved_config)
            elif saved_config is not None:
                config = saved_config
            else:
                n_layer = 8 if "25m" in selected_ckpt.name else (6 if "11m" in selected_ckpt.name else 4)
                n_head = 8 if "25m" in selected_ckpt.name else (6 if "11m" in selected_ckpt.name else 4)
                n_embd = 512 if "25m" in selected_ckpt.name else (384 if "11m" in selected_ckpt.name else 128)
                config = TransformerConfig(
                    vocab_size=tokenizer.vocab_size,
                    block_size=block_size,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_embd=n_embd,
                )
        else:
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
    stop_id = tokenizer.eos_token_id if is_bpe else None
    generated_tensor = model.generate(
        input_tensor,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=25,
        stop_token_id=stop_id,
        repetition_penalty=repetition_penalty,
    )
    generated_ids = generated_tensor[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    for delim in ["<|endoftext|>", "<isendoftextis>"]:
        if delim in generated_text:
            generated_text = generated_text.split(delim)[0].strip()

    print("-" * 50)
    print(generated_text)
    print("-" * 50 + "\n")

def run_chat(
    checkpoint: Optional[str] = None,
    max_tokens: int = 100,
    temperature: float = 0.7,
    repetition_penalty: float = 1.25,
    seed: Optional[int] = None,
):
    """Launch interactive terminal chat session with Robbie."""
    console = Console()
    console.print("\n" + "=" * 60)
    console.print("   PROJECT ROBBIE — INTERACTIVE CHAT (v0.5)")
    console.print("   Type your message and press Enter.")
    console.print("   Type 'exit' or 'quit' to end the session.")
    console.print("=" * 60 + "\n")

    stage2_25m_ckpt = CHECKPOINT_DIR / "stage2_25m_conversational_transformer.pt"
    stage1_25m_ckpt = CHECKPOINT_DIR / "stage1_25m_base_transformer.pt"
    stage2_11m_ckpt = CHECKPOINT_DIR / "stage2_11m_conversational_transformer.pt"
    stage1_11m_ckpt = CHECKPOINT_DIR / "stage1_11m_base_transformer.pt"
    stage2_ckpt = CHECKPOINT_DIR / "stage2_conversational_transformer.pt"
    stage1_ckpt = CHECKPOINT_DIR / "stage1_base_transformer.pt"
    bpe_vocab_file = CHECKPOINT_DIR / "bpe_stage1_vocab.json"

    available_checkpoints = [
        stage2_25m_ckpt,
        stage1_25m_ckpt,
        stage2_11m_ckpt,
        stage1_11m_ckpt,
        stage2_ckpt,
        stage1_ckpt,
    ]

    if checkpoint is not None:
        selected_ckpt = Path(checkpoint)
    else:
        selected_ckpt = next((c for c in available_checkpoints if c.exists()), None)
        if selected_ckpt is None:
            console.print("[red]No trained transformer checkpoint found. Train first with: python -m robbie.main --train-stage2[/]")
            return

    if not bpe_vocab_file.exists():
        console.print(f"[red]BPE vocabulary file not found at {bpe_vocab_file}[/]")
        return

    tokenizer = BPETokenizer.load(bpe_vocab_file)
    raw_data = torch.load(selected_ckpt, map_location="cpu", weights_only=False)
    saved_config = raw_data.get("config", None)
    block_size = getattr(saved_config, "block_size", 128) if saved_config else 128

    if isinstance(saved_config, TransformerConfig):
        config = saved_config
    elif isinstance(saved_config, dict):
        config = TransformerConfig(**saved_config)
    elif saved_config is not None:
        config = saved_config
    else:
        n_layer = 8 if "25m" in selected_ckpt.name else (6 if "11m" in selected_ckpt.name else 4)
        n_head = 8 if "25m" in selected_ckpt.name else (6 if "11m" in selected_ckpt.name else 4)
        n_embd = 512 if "25m" in selected_ckpt.name else (384 if "11m" in selected_ckpt.name else 128)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )
    device = get_best_device()
    model = RobbieTransformer(config)
    Trainer.load_checkpoint(selected_ckpt, model)
    model.to(device)
    model.eval()

    console.print(f"[dim]Loaded model: {selected_ckpt.name} on {str(device).upper()}[/]")
    console.print("[dim]Commands: /history, /memory, /clear, /help, exit/quit[/]\n")

    engine = ChatEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=25,
        repetition_penalty=repetition_penalty,
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session closed. Goodbye![/]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/]")
            break

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        reply = engine.respond(user_input)
        if user_input.strip().startswith("/"):
            console.print(f"[dim]{reply}[/]\n")
        else:
            console.print(f"[bold cyan]Robbie:[/] {reply}\n")


def run_eval_conversation(checkpoint: Optional[str] = None):
    """Run automated Phase 5 Conversational Model quality and multi-turn consistency evaluation."""
    from rich.table import Table

    console = Console()
    console.print("\n" + "=" * 65)
    console.print("   PROJECT ROBBIE — PHASE 5 CONVERSATION EVALUATION")
    console.print("=" * 65 + "\n")

    stage2_25m_ckpt = CHECKPOINT_DIR / "stage2_25m_conversational_transformer.pt"
    stage2_11m_ckpt = CHECKPOINT_DIR / "stage2_11m_conversational_transformer.pt"
    stage2_ckpt = CHECKPOINT_DIR / "stage2_conversational_transformer.pt"
    bpe_vocab_file = CHECKPOINT_DIR / "bpe_stage1_vocab.json"

    available_checkpoints = [stage2_25m_ckpt, stage2_11m_ckpt, stage2_ckpt]

    if checkpoint is not None:
        selected_ckpt = Path(checkpoint)
    else:
        selected_ckpt = next((c for c in available_checkpoints if c.exists()), None)
        if selected_ckpt is None:
            console.print("[red]No stage 2 conversational checkpoint found for evaluation.[/]")
            return

    if not bpe_vocab_file.exists():
        console.print(f"[red]BPE vocabulary file not found at {bpe_vocab_file}[/]")
        return

    tokenizer = BPETokenizer.load(bpe_vocab_file)
    raw_data = torch.load(selected_ckpt, map_location="cpu", weights_only=False)
    saved_config = raw_data.get("config", None)
    block_size = getattr(saved_config, "block_size", 128) if saved_config else 128

    if isinstance(saved_config, TransformerConfig):
        config = saved_config
    elif isinstance(saved_config, dict):
        config = TransformerConfig(**saved_config)
    else:
        n_layer = 8 if "25m" in selected_ckpt.name else 4
        n_head = 8 if "25m" in selected_ckpt.name else 4
        n_embd = 512 if "25m" in selected_ckpt.name else 128
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )

    device = get_best_device()
    model = RobbieTransformer(config)
    Trainer.load_checkpoint(selected_ckpt, model)
    model.to(device)
    model.eval()

    console.print(f"[dim]Loaded model: {selected_ckpt.name} on {str(device).upper()}[/]\n")

    engine = ChatEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_tokens=50,
        temperature=0.4,
    )

    console.print("[yellow]Running multi-turn scenario benchmark...[/]")
    results = ConversationEvaluator.run_multi_turn_test(engine)

    table = Table(title="Phase 5 Multi-Turn Conversation Evaluation")
    table.add_column("Turn", style="dim", width=6)
    table.add_column("Intent", style="cyan", width=18)
    table.add_column("User Input", style="yellow", width=28)
    table.add_column("Robbie Response", style="green", width=42)
    table.add_column("Latency", style="magenta", justify="right")

    for i, t in enumerate(results["turns"], 1):
        table.add_row(
            f"#{i}",
            t["intent"],
            t["user"],
            t["reply"][:40] + ("..." if len(t["reply"]) > 40 else ""),
            f"{t['latency_sec']:.2f}s",
        )

    console.print(table)
    retention_color = "green" if results["name_recalled"] else "red"
    retention_status = "PASSED" if results["name_recalled"] else "FAILED"
    console.print(f"\nName Retention Check: [{retention_color}]{retention_status}[/]")
    console.print(f"Remembered Facts: [bold]{results['slots']}[/]\n")

    # Measure generation throughput
    console.print("[yellow]Measuring generation throughput...[/]")
    speed = ConversationEvaluator.measure_generation_speed(engine)
    console.print(f"Throughput: [bold green]{speed['tokens_per_second']}[/] tokens/sec (Latency: {speed['elapsed_seconds']}s)")
    console.print(f"Distinct-1 N-gram Diversity: [bold cyan]{speed['distinct_1'] * 100:.1f}%[/]")
    console.print(f"Distinct-2 N-gram Diversity: [bold cyan]{speed['distinct_2'] * 100:.1f}%[/]\n")
    console.print("=" * 65 + "\n")

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

    if args.train_stage1:
        run_train_stage1(
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size if args.batch_size != 64 else (32 if args.model_size == "25m" else 64),
            block_size=args.block_size,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            patience=args.patience,
            use_amp=not args.no_amp,
            use_cosine=not args.no_cosine,
        )

    if args.train_stage2:
        run_train_stage2(
            model_size=args.model_size,
            epochs=args.epochs if args.epochs != 5 else 25,
            batch_size=args.batch_size if args.batch_size != 64 else 16,
            block_size=args.block_size,
            stride=args.stride if args.stride is not None else 16,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            patience=args.patience if args.patience is not None else 5,
            use_amp=not args.no_amp,
            use_cosine=not args.no_cosine,
        )

    if args.chat:
        run_chat(
            checkpoint=args.checkpoint,
            max_tokens=args.max_tokens,
            temperature=args.temperature if args.temperature != 0.7 else 0.4,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )

    if args.eval_conversation:
        run_eval_conversation(checkpoint=args.checkpoint)

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
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )

if __name__ == "__main__":
    main()
