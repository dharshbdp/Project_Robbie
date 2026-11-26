"""Fetch pinned Topical-Chat data and attribution from the original repository."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7c939229cbcf6f55f6977b341a5a2f2fe982d53f"
REPOSITORY = "https://github.com/alexa/Topical-Chat"
SPLITS = ("train", "valid_freq", "valid_rare", "test_freq", "test_rare")
FILES = ["README.md", "DATALICENSE", "src/wiki/NOTICE.md", "src/wiki/wiki.json"] + [
    f"{directory}/{split}.json" for directory in ("conversations", "reading_sets/pre-build") for split in SPLITS
]


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fetch(relative, directory):
    path = directory / relative
    if path.is_file():
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    url = f"https://raw.githubusercontent.com/alexa/Topical-Chat/{REVISION}/{relative}"
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Robbie-data-preparation/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as out:
                total, last = 0, time.monotonic()
                while block := response.read(1024 * 1024):
                    out.write(block)
                    total += len(block)
                    if time.monotonic() - last >= 5:
                        print(f"  {relative}: {total / 1e6:.1f} MB", flush=True)
                        last = time.monotonic()
            if path.suffix == ".json":
                json.loads(temporary.read_text(encoding="utf-8"))
            os.replace(temporary, path)
            return path
        except Exception:
            if attempt == 2:
                raise
            print(f"Retrying {relative} ({attempt + 1}/2)", flush=True)
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/chat/Topical-Chat" / REVISION)
    args = parser.parse_args()
    files = []
    for relative in FILES:
        print(f"Topical-Chat: {relative}", flush=True)
        path = fetch(relative, args.output_dir)
        files.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest = {"repository": REPOSITORY, "revision": REVISION, "license": "CDLA-Sharing-1.0",
                "license_exceptions": {"src/wiki/wiki.json": "CC-BY-SA-3.0; see src/wiki/NOTICE.md"},
                "files": files, "note": "Original dataset files; source license and notices preserved. No Reddit credentials or third-party article downloads requested."}
    (args.output_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded {sum(item['bytes'] for item in files) / 1e6:.1f} MB to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
