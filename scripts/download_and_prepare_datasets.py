"""Download and process HuggingFaceH4/no_robots and databricks/databricks-dolly-15k for Project Robbie."""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests
import pandas as pd

RAW_DIR = ROOT / "data/raw/huggingface_datasets"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest_path: Path) -> Path:
    """Download a file with progress reporting."""
    if dest_path.is_file() and dest_path.stat().st_size > 1000:
        print(f"File already exists: {dest_path.name} ({dest_path.stat().st_size / 1e6:.2f} MB)")
        return dest_path

    print(f"Downloading {url} -> {dest_path.name}...")
    headers = {"User-Agent": "ProjectRobbie/1.0"}
    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = (downloaded / total) * 100
                        print(f"\r  {downloaded / 1e6:.2f} MB / {total / 1e6:.2f} MB ({percent:.1f}%)", end="", flush=True)
    print(f"\n  Done! Saved to {dest_path}")
    return dest_path


def load_dolly_15k(file_path: Path):
    """Load and format databricks-dolly-15k into Robbie multi-turn conversation format."""
    conversations = []
    skipped = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            instr = item.get("instruction", "").strip()
            ctx = item.get("context", "").strip()
            resp = item.get("response", "").strip()

            if not instr or not resp:
                skipped += 1
                continue

            if ctx:
                user_msg = f"{instr}\n\nContext:\n{ctx}"
            else:
                user_msg = instr

            # Fit reasonable length for 384-token context budget
            if len(user_msg) > 1200 or len(resp) > 1000:
                # Truncate or take reasonable length
                user_msg = user_msg[:1200]
                resp = resp[:1000]

            conv = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": resp}
            ]
            conversations.append({
                "conversation": conv,
                "source": "databricks_dolly_15k",
                "category": item.get("category", "general")
            })

    print(f"Loaded {len(conversations)} conversations from Dolly-15k (skipped {skipped})")
    return conversations


def load_no_robots(parquet_path: Path):
    """Load and format HuggingFaceH4/no_robots parquet into Robbie format."""
    df = pd.read_parquet(parquet_path)
    print(f"no_robots columns: {list(df.columns)}, row count: {len(df)}")
    conversations = []

    for _, row in df.iterrows():
        raw_msgs = row.get("messages")
        if raw_msgs is not None:
            msgs = list(raw_msgs)
            if len(msgs) >= 2:
                clean_conv = []
                for m in msgs:
                    if isinstance(m, dict):
                        role = m.get("role", "")
                        content = m.get("content", "").strip()
                        if role in ("user", "assistant") and content:
                            clean_conv.append({"role": role, "content": content[:1200]})
                if len(clean_conv) >= 2 and clean_conv[0]["role"] == "user":
                    conversations.append({
                        "conversation": clean_conv,
                        "source": "huggingface_no_robots",
                        "category": row.get("category", "instruction")
                    })
        elif "prompt" in row and "messages" not in row:
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("response", "")).strip()
            if prompt and response:
                conversations.append({
                    "conversation": [
                        {"role": "user", "content": prompt[:1200]},
                        {"role": "assistant", "content": response[:1000]}
                    ],
                    "source": "huggingface_no_robots",
                    "category": row.get("category", "instruction")
                })

    print(f"Loaded {len(conversations)} conversations from no_robots")
    return conversations


def main():
    print("=" * 70)
    print("Downloading HuggingFaceH4/no_robots & databricks/databricks-dolly-15k")
    print("=" * 70)

    # 1. Dolly 15k
    dolly_url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
    dolly_path = RAW_DIR / "databricks-dolly-15k.jsonl"
    download_file(dolly_url, dolly_path)
    dolly_convs = load_dolly_15k(dolly_path)

    # 2. No Robots
    norobots_train_url = "https://huggingface.co/datasets/HuggingFaceH4/no_robots/resolve/main/data/train-00000-of-00001.parquet"
    norobots_path = RAW_DIR / "no_robots_train.parquet"
    download_file(norobots_train_url, norobots_path)
    norobots_convs = load_no_robots(norobots_path)

    # Combine into unified dataset
    combined = dolly_convs + norobots_convs
    out_file = ROOT / "data/curated/dolly_and_norobots_curated.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nUnified curated dataset written to:\n  {out_file}")
    print(f"Total conversations: {len(combined)} ({len(dolly_convs)} Dolly-15k + {len(norobots_convs)} No Robots)")


if __name__ == "__main__":
    main()
