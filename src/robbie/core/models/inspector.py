"""Model inspection, parameter auditing, and baseline evaluation tools for Project Robbie."""

from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
from rich.table import Table
from rich.console import Console

from robbie.utils.logger import logger
from robbie.core.models.tiny_llm import TinyLLM, TinyLLMConfig
from robbie.core.tokenizer import BaseTokenizer


def get_model_parameter_summary(model: nn.Module) -> Dict[str, Any]:
    """Calculate parameter breakdown for every named layer in the model."""
    layer_info = []
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

        layer_info.append({
            "name": name,
            "shape": list(param.shape),
            "parameters": num_params,
            "trainable": param.requires_grad,
        })

    return {
        "layers": layer_info,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }


def print_architecture_inspection(model: TinyLLM):
    """Print an architectural and parameter breakdown table."""
    summary = get_model_parameter_summary(model)
    console = Console()

    table = Table(title="TinyLLM (v0.2) Architecture & Parameter Inspection")
    table.add_column("Layer / Module", style="cyan", no_wrap=True)
    table.add_column("Tensor Shape", style="magenta")
    table.add_column("Parameter Count", justify="right", style="green")
    table.add_column("Trainable", justify="center", style="yellow")

    for layer in summary["layers"]:
        table.add_row(
            layer["name"],
            str(layer["shape"]),
            f"{layer['parameters']:,}",
            "Yes" if layer["trainable"] else "No",
        )

    console.print(table)
    console.print(f"[bold green]Total Parameters:[/] {summary['total_parameters']:,}")
    console.print(f"[bold green]Trainable Parameters:[/] {summary['trainable_parameters']:,}")
    console.print(f"[bold blue]Context Window (block_size):[/] {model.config.block_size} tokens")
    console.print(f"[bold blue]Embedding Dimension (n_embd):[/] {model.config.n_embd}")
    console.print(f"[bold blue]Hidden Dimension (n_hidden):[/] {model.config.n_hidden}")
    console.print(f"[bold blue]Vocabulary Size:[/] {model.config.vocab_size} tokens\n")


def inspect_dataset_alignment(
    text: str,
    tokenizer: BaseTokenizer,
    seq_len: int = 16,
) -> None:
    """Demonstrate exact tokenization and (x, y) next-token offset alignment."""
    console = Console()
    console.print("[bold yellow]=== Dataset & Next-Token Alignment Inspection ===[/]\n")

    sample_snippet = text[:100].replace("\n", "\\n")
    console.print(f"[bold]Raw Sample Text (first 100 chars):[/]\n\"{sample_snippet}\"\n")

    tokens = tokenizer.encode(text[:200])
    x_tokens = tokens[:seq_len]
    y_tokens = tokens[1 : seq_len + 1]

    console.print(f"[bold cyan]Input  Tokens (x) [len={len(x_tokens)}]:[/] {x_tokens}")
    console.print(f"[bold green]Target Tokens (y) [len={len(y_tokens)}]:[/] {y_tokens}\n")

    # Show token by token prediction target
    table = Table(title="Step-by-Step Next-Token Prediction Offset")
    table.add_column("Step t", justify="right", style="cyan")
    table.add_column("Input Token ID (x[t])", justify="right", style="magenta")
    table.add_column("Input Char", style="yellow")
    table.add_column("Target Token ID (y[t])", justify="right", style="magenta")
    table.add_column("Target Char (Predict Next)", style="green")

    for t in range(min(seq_len, 10)):
        x_char = tokenizer.decode([x_tokens[t]]).replace("\n", "\\n")
        y_char = tokenizer.decode([y_tokens[t]]).replace("\n", "\\n")
        table.add_row(
            str(t),
            str(x_tokens[t]),
            repr(x_char),
            str(y_tokens[t]),
            repr(y_char),
        )

    console.print(table)
    console.print("Notice how for every step t, target y[t] is precisely the character immediately following x[t].\n")


def benchmark_generation(
    model: TinyLLM,
    tokenizer: BaseTokenizer,
    prompts: List[str],
    max_tokens: int = 60,
    temperature: float = 0.8,
    seed: int = 42,
    device: str = "cpu",
) -> None:
    """Generate completions across standardized benchmark prompts for comparison."""
    torch.manual_seed(seed)
    console = Console()
    console.print(f"[bold yellow]=== Standardized Benchmark Generation (Temp: {temperature}, Seed: {seed}) ===[/]\n")

    model.to(device)
    model.eval()

    for idx, prompt in enumerate(prompts, 1):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            prompt_ids = [tokenizer.bos_token_id]
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            generated_tensor = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

        gen_text = tokenizer.decode(generated_tensor[0].tolist(), skip_special_tokens=True)
        console.print(f"[bold cyan]Benchmark Prompt {idx}:[/] [bold]\"{prompt}\"[/]")
        console.print(f"[green]{gen_text}[/]\n" + "-" * 50)
