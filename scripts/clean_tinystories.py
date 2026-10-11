"""Clean and validate TinyStories dataset."""
from pathlib import Path

def clean_tinystories():
    p = Path("data/raw/stage1_pretrain/tinystories.txt")
    if not p.exists():
        print("tinystories.txt does not exist.")
        return
    text = p.read_text(encoding="utf-8", errors="replace")
    tag = "<|endoftext|>"
    last_idx = text.rfind(tag)
    if last_idx != -1:
        clean_text = text[:last_idx + len(tag)].strip()
        p.write_text(clean_text, encoding="utf-8")
        num_stories = clean_text.count(tag)
        print(f"Cleaned tinystories.txt: {len(clean_text):,} chars, {num_stories:,} complete stories.")
    else:
        print("No delimiter found.")

if __name__ == "__main__":
    clean_tinystories()
