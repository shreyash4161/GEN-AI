from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reporting import generate_report

parser = argparse.ArgumentParser(description="Generate a report from saved run.json records; missing scores stay blank.")
parser.add_argument("--runs-dir", type=Path, default=ROOT / "results" / "models")
parser.add_argument("--output", type=Path, default=ROOT / "results" / "reports" / "FINAL_RESEARCH_REPORT.md")
args = parser.parse_args()
markdown, csv = generate_report(args.runs_dir, args.output)
print(f"Markdown report: {markdown}\nMetrics CSV: {csv}")
