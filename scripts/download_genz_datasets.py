"""Acquire, clean, and convert Gen-Z slang and conversational datasets for Project Robbie."""

import csv
import io
import json
from pathlib import Path
import urllib.request
import zipfile
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.table import Table

from robbie.config import DATA_DIR
from robbie.utils.logger import logger


URL_MLBTRIO = "https://huggingface.co/datasets/MLBtrio/genz-slang-dataset/resolve/main/all_slangs.csv"
URL_PROGRAMMER = "https://huggingface.co/datasets/Programmer-RD-AI/genz-slang-pairs-1k/resolve/main/genz_dataset.csv"
URL_GCRUZ = "https://huggingface.co/datasets/GCruz19/GenZ_data/resolve/main/genz%20(3).json"
URL_KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/tawfiayeasmin/gen-z-words-and-phrases-dataset"


def fetch_mlbtrio(console: Optional[Console] = None) -> List[Dict[str, Any]]:
    """Fetch MLBtrio/genz-slang-dataset (1,779 slangs) and format into dialogues."""
    if console:
        console.print("[cyan]Fetching MLBtrio/genz-slang-dataset from HuggingFace...[/]")

    req = urllib.request.Request(URL_MLBTRIO, headers={"User-Agent": "Mozilla/5.0"})
    dialogues = []

    with urllib.request.urlopen(req, timeout=20) as r:
        reader = csv.DictReader(io.StringIO(r.read().decode("utf-8", errors="replace")))
        for row in reader:
            slang = row.get("Slang", "").strip()
            desc = row.get("Description", "").strip()
            example = row.get("Example", "").strip().strip('"')

            if not slang or not desc:
                continue

            # Variation 1: Definition inquiry
            ex_str = f' For example: "{example}".' if example else ""
            dialogues.append({
                "source": "MLBtrio/genz-slang-dataset",
                "conversation": [
                    {"role": "user", "content": f"What does '{slang}' mean?"},
                    {"role": "assistant", "content": f"'{slang}' means {desc}.{ex_str}"},
                ],
            })

            # Variation 2: Usage example inquiry
            if example:
                dialogues.append({
                    "source": "MLBtrio/genz-slang-dataset",
                    "conversation": [
                        {"role": "user", "content": f"How do people use the slang '{slang}' in a sentence?"},
                        {"role": "assistant", "content": f'People often say: "{example}"! It means {desc.lower()}.'},
                    ],
                })

    return dialogues


def fetch_programmer_rd(console: Optional[Console] = None) -> List[Dict[str, Any]]:
    """Fetch Programmer-RD-AI/genz-slang-pairs-1k (1,005 pairs) and format into dialogues."""
    if console:
        console.print("[cyan]Fetching Programmer-RD-AI/genz-slang-pairs-1k from HuggingFace...[/]")

    req = urllib.request.Request(URL_PROGRAMMER, headers={"User-Agent": "Mozilla/5.0"})
    dialogues = []

    with urllib.request.urlopen(req, timeout=20) as r:
        reader = csv.DictReader(io.StringIO(r.read().decode("utf-8", errors="replace")))
        for row in reader:
            normal = row.get("normal", "").strip().strip('"')
            genz = row.get("gen_z", "").strip().strip('"')

            if not normal or not genz:
                continue

            # Variation 1: Translation query
            dialogues.append({
                "source": "Programmer-RD-AI/genz-slang-pairs-1k",
                "conversation": [
                    {"role": "user", "content": f'How would you say this in Gen-Z slang: "{normal}"?'},
                    {"role": "assistant", "content": f'In Gen-Z slang, you\'d say: "{genz}"!'},
                ],
            })

            # Variation 2: Conversational response
            dialogues.append({
                "source": "Programmer-RD-AI/genz-slang-pairs-1k",
                "conversation": [
                    {"role": "user", "content": normal},
                    {"role": "assistant", "content": genz},
                ],
            })

    return dialogues


def fetch_gcruz(console: Optional[Console] = None) -> List[Dict[str, Any]]:
    """Fetch GCruz19/GenZ_data (821 pairs) and format into dialogues."""
    if console:
        console.print("[cyan]Fetching GCruz19/GenZ_data from HuggingFace...[/]")

    req = urllib.request.Request(URL_GCRUZ, headers={"User-Agent": "Mozilla/5.0"})
    dialogues = []

    with urllib.request.urlopen(req, timeout=20) as r:
        raw_items = json.loads(r.read().decode("utf-8", errors="replace"))

    for item in raw_items:
        trans_data = item.get("translation")
        if isinstance(trans_data, str):
            try:
                trans = json.loads(trans_data)
            except Exception:
                continue
        elif isinstance(trans_data, dict):
            trans = trans_data
        else:
            continue

        normal = trans.get("normal", "").strip().strip('"')
        genz = trans.get("genz", "").strip().strip('"')

        if not normal or not genz:
            continue

        dialogues.append({
            "source": "GCruz19/GenZ_data",
            "conversation": [
                {"role": "user", "content": f'How would Gen-Z rephrase: "{normal}"?'},
                {"role": "assistant", "content": f'That would be: "{genz}"!'},
            ],
        })

    return dialogues


def fetch_kaggle_genz(console: Optional[Console] = None) -> List[Dict[str, Any]]:
    """Fetch tawfiayeasmin/gen-z-words-and-phrases-dataset from Kaggle and format into dialogues."""
    if console:
        console.print("[cyan]Fetching tawfiayeasmin/gen-z-words-and-phrases-dataset from Kaggle...[/]")

    req = urllib.request.Request(URL_KAGGLE, headers={"User-Agent": "Mozilla/5.0"})
    dialogues = []

    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            zf = zipfile.ZipFile(io.BytesIO(r.read()))
            csv_text = zf.read("gen_zz_words.csv").decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(csv_text))

            for row in reader:
                word = row.get("Word/Phrase", "").strip()
                defn = row.get("Definition", "").strip()
                example = row.get("Example Sentence", "").strip().strip('"')
                trend = row.get("Popularity/Trend Level", "").strip()

                if not word or not defn:
                    continue

                ex_part = f' For example: "{example}".' if example else ""
                trend_part = f" ({trend} popularity)" if trend else ""

                dialogues.append({
                    "source": "tawfiayeasmin/gen-z-words-and-phrases-dataset",
                    "conversation": [
                        {"role": "user", "content": f"What does '{word}' mean in Gen-Z slang?"},
                        {"role": "assistant", "content": f"'{word}'{trend_part} means: {defn}.{ex_part}"},
                    ],
                })
    except Exception as e:
        logger.warning(f"Failed to fetch Kaggle dataset: {e}")

    return dialogues


def main():
    console = Console()
    console.print("\n" + "=" * 65)
    console.print("       PROJECT ROBBIE — GEN-Z SLANG DATASET IMPORTER")
    console.print("=" * 65 + "\n")

    all_dialogues = []
    stats = {}

    with console.status("[bold cyan]Fetching and formatting all 4 Gen-Z datasets...[/]"):
        d1 = fetch_mlbtrio(console=console)
        stats["MLBtrio/genz-slang-dataset"] = len(d1)
        all_dialogues.extend(d1)

        d2 = fetch_programmer_rd(console=console)
        stats["Programmer-RD-AI/genz-slang-pairs-1k"] = len(d2)
        all_dialogues.extend(d2)

        d3 = fetch_gcruz(console=console)
        stats["GCruz19/GenZ_data"] = len(d3)
        all_dialogues.extend(d3)

        d4 = fetch_kaggle_genz(console=console)
        stats["tawfiayeasmin/gen-z-words-and-phrases (Kaggle)"] = len(d4)
        all_dialogues.extend(d4)

    # Save formatted dialogues to data/raw/custom/genz_slang_dialogues.jsonl
    output_path = DATA_DIR / "raw" / "custom" / "genz_slang_dialogues.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for d in all_dialogues:
            f.write(json.dumps(d) + "\n")

    table = Table(title="Imported Gen-Z Datasets")
    table.add_column("Dataset Source", style="bold yellow")
    table.add_column("Generated Dialogues", style="green", justify="right")

    for name, cnt in stats.items():
        table.add_row(name, f"{cnt:,}")

    console.print(table)
    console.print(f"\n[bold green]Successfully saved {len(all_dialogues):,} Gen-Z dialogues to:[/] `{output_path.relative_to(DATA_DIR)}`\n")


if __name__ == "__main__":
    main()
