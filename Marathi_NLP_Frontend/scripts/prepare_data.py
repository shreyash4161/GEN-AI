"""Download and validate the official four-class L3Cube-MahaHate XLSX splits."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_core import LABELS, load_dataset

BASE_URL = "https://raw.githubusercontent.com/l3cube-pune/MarathiNLP/main/L3Cube-MahaHate/4-class"
DATA_DIR = ROOT / "data" / "maha_hate"


def download(name: str) -> Path:
    target = DATA_DIR / f"hate_{name}.xlsx"
    if target.exists():
        return target
    temporary = target.with_suffix(target.suffix + ".part")
    response = requests.get(f"{BASE_URL}/{target.name}", timeout=(15, 120), stream=True)
    response.raise_for_status()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as stream:
        for chunk in response.iter_content(128 * 1024):
            if chunk:
                stream.write(chunk)
    if temporary.stat().st_size < 1024:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Downloaded {target.name} is unexpectedly small")
    temporary.replace(target)
    return target


def main() -> int:
    summary = {}
    for split in ("train", "valid", "test"):
        path = download(split)
        raw = pd.read_excel(path)
        frame, metadata = load_dataset(path)
        if set(frame.label) != set(LABELS):
            raise ValueError(f"{split} does not contain all four target classes: {metadata['label_counts']}")
        summary[split] = {
            "file": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "raw_rows": len(raw), "validated_rows": len(frame),
            "label_counts": metadata["label_counts"],
        }
    (DATA_DIR / "dataset_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
