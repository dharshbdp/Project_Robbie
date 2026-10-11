"""Download starter training datasets for Project Robbie."""

import urllib.request
from pathlib import Path
import sys

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEST_FILE = DATA_DIR / "tiny_shakespeare.txt"

FALLBACK_TEXT = """ROBBIE:
Hello. I am Robbie, your personal artificial intelligence assistant.
I am learning to understand language, token by token, from the ground up.
Neural networks learn statistical patterns in sequences of numbers.
By predicting the next token over many iterations, I develop intuition for grammar, structure, and meaning.
Today I am a tiny language model. Tomorrow I will be a complete assistant with memory, tools, and computer awareness.
""" * 500

def download_dataset():
    """Download the TinyShakespeare dataset or generate local fallback if offline."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DEST_FILE.exists() and DEST_FILE.stat().st_size > 0:
        print(f"[Dataset] Already present at: {DEST_FILE} ({DEST_FILE.stat().st_size / 1024:.1f} KB)")
        return DEST_FILE

    print(f"[Dataset] Downloading starter dataset from:\n  {DATA_URL}")
    try:
        urllib.request.urlretrieve(DATA_URL, DEST_FILE)
        size_kb = DEST_FILE.stat().st_size / 1024
        print(f"[Dataset] Successfully downloaded to {DEST_FILE} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"[Dataset] Download failed ({e}). Writing local starter corpus instead...")
        with open(DEST_FILE, "w", encoding="utf-8") as f:
            f.write(FALLBACK_TEXT)
        print(f"[Dataset] Wrote fallback corpus to {DEST_FILE} ({DEST_FILE.stat().st_size / 1024:.1f} KB)")

    return DEST_FILE

if __name__ == "__main__":
    download_dataset()
