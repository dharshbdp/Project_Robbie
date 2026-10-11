"""CLI script to acquire, clean, and prepare datasets for Project Robbie."""

import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

from robbie.core.data import DatasetPipeline
from robbie.config import DATA_DIR


def print_stage_stats(meta_file: Path, console: Console):
    """Print formatted summary table from stage metadata JSON."""
    if not meta_file.exists():
        console.print(f"[yellow]Metadata file not found at: {meta_file}[/]")
        return

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    stage_num = meta.get("stage", "?")
    name = meta.get("name", "Unknown Stage")
    desc = meta.get("description", "")
    train = meta.get("train", {})
    val = meta.get("val", {})

    table = Table(title=f"Stage {stage_num} Dataset: {name}")
    table.add_column("Split", style="cyan", justify="left")
    table.add_column("Characters", style="green", justify="right")
    table.add_column("Words", style="yellow", justify="right")
    table.add_column("Estimated Tokens", style="magenta", justify="right")

    table.add_row(
        "Train",
        f"{train.get('character_count', 0):,}",
        f"{train.get('word_count', 0):,}",
        f"{train.get('estimated_tokens', 0):,}",
    )
    table.add_row(
        "Validation",
        f"{val.get('character_count', 0):,}",
        f"{val.get('word_count', 0):,}",
        f"{val.get('estimated_tokens', 0):,}",
    )

    console.print(table)
    console.print(f"[dim]{desc}[/]\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare Project Robbie training datasets.")
    parser.add_argument("--stage", type=int, choices=[1, 2], help="Which stage to prepare (1: Pretrain, 2: Dialogue)")
    parser.add_argument("--all", action="store_true", help="Prepare all training stages")
    parser.add_argument("--stats", action="store_true", help="Display statistics for prepared datasets")
    args = parser.parse_args()

    console = Console()
    pipeline = DatasetPipeline(data_root=DATA_DIR)

    if args.stage == 1 or args.all:
        console.print("[bold cyan]=== Preparing Stage 1 (Base Model Pretraining) ===[/]")
        meta1 = pipeline.prepare_stage1_pretrain()
        print_stage_stats(DATA_DIR / "processed" / "stage1" / "dataset_metadata.json", console)

    if args.stage == 2 or args.all:
        console.print("[bold cyan]=== Preparing Stage 2 (Conversational Fine-Tuning) ===[/]")
        meta2 = pipeline.prepare_stage2_dialogue(force=True)
        print_stage_stats(DATA_DIR / "processed" / "stage2" / "dataset_metadata.json", console)

    if args.stats and not args.stage and not args.all:
        console.print("\n[bold]=== Project Robbie Dataset Registry ===[/]\n")
        print_stage_stats(DATA_DIR / "processed" / "stage1" / "dataset_metadata.json", console)
        print_stage_stats(DATA_DIR / "processed" / "stage2" / "dataset_metadata.json", console)

    if not args.stage and not args.all and not args.stats:
        parser.print_help()


if __name__ == "__main__":
    main()
