"""Side-by-side comparative text generation between model checkpoints."""

import argparse
from pathlib import Path
from typing import List, Optional
import torch

from robbie.core.tokenizer.char_tokenizer import CharTokenizer
from robbie.core.models.transformer import RobbieTransformer, TransformerConfig
from robbie.core.training.trainer import Trainer
from robbie.utils.device import get_best_device


DEFAULT_PROMPTS = [
    "ROBBIE: ",
    "To be or not to be",
    "First Citizen: ",
]


def generate_sample(
    model: RobbieTransformer,
    tokenizer: CharTokenizer,
    prompt: str,
    max_tokens: int = 70,
    temperature: float = 0.8,
    seed: int = 42,
    device: str = "cpu",
) -> str:
    """Generate completion with deterministic seed and sampling settings."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_token_id]
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_tensor = model.generate(
        input_tensor,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=20,
    )
    generated_ids = generated_tensor[0].tolist()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def compare_checkpoints(
    ckpt_a: Path,
    ckpt_b: Path,
    label_a: str = "Checkpoint A",
    label_b: str = "Checkpoint B",
    vocab_file: Path = Path("checkpoints/transformer_vocab.json"),
    prompts: Optional[List[str]] = None,
    temperature: float = 0.8,
    max_tokens: int = 70,
    seed: int = 42,
):
    prompts = prompts or DEFAULT_PROMPTS
    device = get_best_device()

    tokenizer = CharTokenizer.load(vocab_file)
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128,
    )

    # Load Model A
    model_a = RobbieTransformer(config).to(device)
    meta_a = Trainer.load_checkpoint(ckpt_a, model_a)
    model_a.eval()

    # Load Model B
    model_b = RobbieTransformer(config).to(device)
    meta_b = Trainer.load_checkpoint(ckpt_b, model_b)
    model_b.eval()

    val_a = meta_a.get("val_loss", "N/A")
    val_b = meta_b.get("val_loss", "N/A")
    epoch_a = meta_a.get("epoch", "N/A")
    epoch_b = meta_b.get("epoch", "N/A")

    print("\n" + "=" * 70)
    print("      PROJECT ROBBIE — COMPARATIVE CHECKPOINT BENCHMARK")
    print("=" * 70)
    print(f"Sampling Settings: Temperature={temperature} | MaxTokens={max_tokens} | Seed={seed}")
    loss_str_a = f"{val_a:.4f}" if isinstance(val_a, (int, float)) else str(val_a)
    loss_str_b = f"{val_b:.4f}" if isinstance(val_b, (int, float)) else str(val_b)
    print(f"[{label_a}]: {ckpt_a.name} (Epoch {epoch_a}, Val Loss: {loss_str_a})")
    print(f"[{label_b}]: {ckpt_b.name} (Epoch {epoch_b}, Val Loss: {loss_str_b})")
    print("=" * 70 + "\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: \"{prompt}\"")
        print("-" * 70)

        sample_a = generate_sample(model_a, tokenizer, prompt, max_tokens, temperature, seed, device)
        sample_b = generate_sample(model_b, tokenizer, prompt, max_tokens, temperature, seed, device)

        print(f"--- {label_a} (Epoch {epoch_a}) ---")
        print(sample_a.strip())
        print(f"\n--- {label_b} (Epoch {epoch_b}) ---")
        print(sample_b.strip())
        print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare text generation between two checkpoints.")
    parser.add_argument("--ckpt-a", type=str, default="checkpoints/transformer_epoch3_best.pt")
    parser.add_argument("--ckpt-b", type=str, default="checkpoints/transformer_epoch5.pt")
    parser.add_argument("--label-a", type=str, default="Epoch 3 (Best Val Loss)")
    parser.add_argument("--label-b", type=str, default="Epoch 5 (Final)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=70)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    compare_checkpoints(
        ckpt_a=Path(args.ckpt_a),
        ckpt_b=Path(args.ckpt_b),
        label_a=args.label_a,
        label_b=args.label_b,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
