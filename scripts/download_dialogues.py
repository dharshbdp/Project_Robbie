"""Download and prepare high-quality open-source dialogue datasets for Project Robbie."""

import argparse
import json
from pathlib import Path
import re
import urllib.request
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.table import Table

from robbie.config import DATA_DIR
from robbie.utils.logger import logger


ALPACA_URL = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
DOLLY_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"


def download_alpaca_dialogues(
    output_path: Path,
    limit: int = 3000,
    max_words: int = 140,
    console: Optional[Console] = None,
) -> int:
    """Download and filter concise instruction-response pairs from Cleaned Alpaca."""
    if console:
        console.print(f"[bold cyan]Downloading Cleaned Alpaca dataset from GitHub...[/]")

    req = urllib.request.Request(ALPACA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw_data = json.loads(resp.read().decode("utf-8"))

    filtered_dialogues = []
    for item in raw_data:
        instruction = item.get("instruction", "").strip()
        user_input = item.get("input", "").strip()
        output = item.get("output", "").strip()

        if not instruction or not output:
            continue

        # Combine instruction + input into user query
        user_content = f"{instruction}\n{user_input}".strip() if user_input else instruction

        # Filter out overly lengthy examples that exceed Robbie's context window
        total_words = len(user_content.split()) + len(output.split())
        if total_words > max_words or total_words < 6:
            continue

        # Skip markdown code blocks that are too long
        if output.count("```") > 2:
            continue

        record = {
            "source": "alpaca_cleaned",
            "conversation": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ],
        }
        filtered_dialogues.append(record)
        if len(filtered_dialogues) >= limit:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for d in filtered_dialogues:
            f.write(json.dumps(d) + "\n")

    return len(filtered_dialogues)


def download_dolly_dialogues(
    output_path: Path,
    limit: int = 3000,
    max_words: int = 140,
    console: Optional[Console] = None,
) -> int:
    """Download and filter conversational instructions from Databricks Dolly-15k."""
    if console:
        console.print(f"[bold cyan]Downloading Databricks Dolly-15k from HuggingFace...[/]")

    req = urllib.request.Request(DOLLY_URL, headers={"User-Agent": "Mozilla/5.0"})
    filtered_dialogues = []

    with urllib.request.urlopen(req, timeout=30) as resp:
        for line in resp:
            if not line.strip():
                continue
            try:
                item = json.loads(line.decode("utf-8"))
            except Exception:
                continue

            instruction = item.get("instruction", "").strip()
            context = item.get("context", "").strip()
            response = item.get("response", "").strip()

            if not instruction or not response:
                continue

            user_content = f"{instruction}\n{context}".strip() if context else instruction
            total_words = len(user_content.split()) + len(response.split())

            if total_words > max_words or total_words < 6:
                continue

            record = {
                "source": "databricks_dolly_15k",
                "category": item.get("category", "general"),
                "conversation": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": response},
                ],
            }
            filtered_dialogues.append(record)
            if len(filtered_dialogues) >= limit:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for d in filtered_dialogues:
            f.write(json.dumps(d) + "\n")

    return len(filtered_dialogues)


def main():
    parser = argparse.ArgumentParser(description="Download high-quality dialogue datasets for Robbie.")
    parser.add_argument(
        "--source",
        choices=["alpaca", "dolly", "all"],
        default="all",
        help="Which dialogue dataset source to fetch (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2500,
        help="Max number of filtered dialogues to retain per source (default: 2500)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=120,
        help="Maximum total words (user + assistant) to fit context window (default: 120)",
    )
    args = parser.parse_args()

    console = Console()
    console.print("\n" + "=" * 65)
    console.print("       PROJECT ROBBIE — DIALOGUE DATASET ACQUISITION")
    console.print("=" * 65 + "\n")

    stage2_raw = DATA_DIR / "raw" / "stage2_dialogue"
    results = {}

    if args.source in ("alpaca", "all"):
        alpaca_file = stage2_raw / "alpaca_clean_dialogues.jsonl"
        with console.status("[bold cyan]Fetching and filtering Cleaned Alpaca dialogues...[/]"):
            count = download_alpaca_dialogues(
                alpaca_file,
                limit=args.limit,
                max_words=args.max_words,
                console=console,
            )
        results["Cleaned Alpaca (Instruction & Q&A)"] = (count, alpaca_file)

    if args.source in ("dolly", "all"):
        dolly_file = stage2_raw / "dolly_dialogues.jsonl"
        with console.status("[bold cyan]Fetching and filtering Databricks Dolly-15k dialogues...[/]"):
            count = download_dolly_dialogues(
                dolly_file,
                limit=args.limit,
                max_words=args.max_words,
                console=console,
            )
        results["Databricks Dolly-15k (Conversational)"] = (count, dolly_file)

    table = Table(title="Acquired Dialogue Datasets")
    table.add_column("Dataset Source", style="bold yellow")
    table.add_column("Dialogues", style="green", justify="right")
    table.add_column("Target File", style="cyan")

    total_added = 0
    for name, (count, fpath) in results.items():
        table.add_row(name, f"{count:,}", str(fpath.name))
        total_added += count

    console.print(table)
    console.print(f"\n[bold green]Successfully acquired {total_added:,} new dialogue pairs![/]")
    console.print("[dim]Next: Drop any custom files into 'data/raw/custom/' then run:[/]")
    console.print("[bold cyan]python scripts/prepare_data.py --stage 2[/]\n")


if __name__ == "__main__":
    main()
