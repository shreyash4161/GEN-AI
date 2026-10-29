"""Preview or serially execute a deliberately bounded A0-A6 run matrix."""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("MahaBERT", "MuRIL", "XLM-RoBERTa", "IndicBERT v2")
ARMS = tuple(f"A{i}" for i in range(7))
PROPORTIONS = (0.05, 0.10, 0.25, 0.50, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["MahaBERT"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["A0"])
    parser.add_argument("--proportions", nargs="+", type=float, choices=PROPORTIONS, default=[0.05])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--synthetic-jsonl", type=Path, default=None)
    parser.add_argument("--synthetic-budget", type=int, default=0)
    parser.add_argument("--resumable-checkpoints", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Without this flag, print the plan only")
    parser.add_argument("--acknowledge-large-matrix", action="store_true", help="Required to execute more than 6 runs")
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > 5 or not 1 <= args.batch_size <= 4:
        parser.error("For this launcher, epochs must be 1-5 and batch-size 1-4")
    if any(arm != "A0" for arm in args.arms) and (not args.synthetic_jsonl or args.synthetic_budget < 1):
        parser.error("A1-A6 require --synthetic-jsonl and a positive --synthetic-budget")
    if args.synthetic_jsonl and not args.synthetic_jsonl.is_absolute():
        args.synthetic_jsonl = (ROOT / args.synthetic_jsonl).resolve()
    plan = list(itertools.product(args.models, args.arms, args.proportions))
    print(f"Planned runs: {len(plan)}")
    for index, (model, arm, proportion) in enumerate(plan, 1):
        print(f"{index:02d}. {model} / {arm} / real-data={proportion:.0%} / seed={args.seed} / epochs={args.epochs}")
    if not args.execute:
        print("Preview only. Add --execute to launch sequential training. No model or API work was started.")
        return 0
    if len(plan) > 6 and not args.acknowledge_large_matrix:
        parser.error("More than six runs may take a long time and use substantial disk. Add --acknowledge-large-matrix after reviewing the plan.")
    if len(args.models) > 1 and any(arm != "A0" for arm in args.arms):
        parser.error("For a fair model/augmentation comparison, run one arm at a time per model and reuse the same candidate file/budget.")
    trainer = ROOT / "scripts" / "train_model.py"
    for model, arm, proportion in plan:
        command = [sys.executable, str(trainer), "--model", model, "--experiment-arm", arm,
                   "--proportion", str(proportion), "--seed", str(args.seed),
                   "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--evaluate-test"]
        if arm != "A0":
            command += ["--synthetic-jsonl", str(args.synthetic_jsonl), "--synthetic-budget", str(args.synthetic_budget)]
        if args.resumable_checkpoints:
            command.append("--resumable-checkpoints")
        print("Executing:", subprocess.list2cmdline(command), flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            print(f"Run failed with exit code {result.returncode}; later combinations were not launched.", file=sys.stderr)
            return result.returncode
    print("All planned commands finished. Confirm each run.json and registry status before reporting results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
