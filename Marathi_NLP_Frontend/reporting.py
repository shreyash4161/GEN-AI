"""Build a factual Markdown report and CSV from saved run metadata only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def collect_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    runs_dir = Path(runs_dir)
    records = []
    if not runs_dir.exists():
        return records
    for path in sorted(runs_dir.rglob("run.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            item["run_metadata_path"] = str(path)
            records.append(item)
        except (OSError, json.JSONDecodeError) as exc:
            records.append({"experiment_id": path.parent.name, "status": "FAILED", "error": f"Could not parse {path}: {exc}", "run_metadata_path": str(path)})
    # Include failed/interrupted attempts, which may have no run.json because
    # the process stopped before final metadata was written.
    registry_path = runs_dir.parent / "experiment_registry.csv"
    if registry_path.exists():
        known = {str(item.get("experiment_id")) for item in records}
        try:
            for row in pd.read_csv(registry_path).to_dict("records"):
                experiment_id = str(row.get("experiment_id", ""))
                if experiment_id and experiment_id not in known:
                    records.append({**row, "run_metadata_path": "registry only; no finalized run.json"})
        except (OSError, pd.errors.ParserError):
            pass
    return records


def collect_live_candidates(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Read only persisted API candidate logs; smoke fixtures live elsewhere."""
    generated_dir = Path(runs_dir).parent / "generated_data"
    records: list[dict[str, Any]] = []
    if not generated_dir.exists():
        return records
    for path in sorted(generated_dir.glob("*_candidates.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                metadata = item.get("generation_metadata") or {}
                if metadata.get("provider") == "openai_compatible":
                    records.append(item)
        except (OSError, json.JSONDecodeError):
            continue
    return records


def generate_report(runs_dir: str | Path, output_path: str | Path) -> tuple[Path, Path]:
    runs = collect_runs(runs_dir)
    candidates = collect_live_candidates(runs_dir)
    accepted_candidates = sum(bool(item.get("accepted")) for item in candidates)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in runs:
        validation = run.get("validation_metrics") or {}
        test = run.get("test_metrics") or {}
        rows.append({
            "experiment_id": run.get("experiment_id", "unknown"), "status": run.get("status", "UNKNOWN"),
            "model": run.get("model"), "arm": run.get("experiment_arm", "A0"), "seed": run.get("seed"),
            "proportion": run.get("effective_proportion", run.get("proportion")),
            "real_train_rows": run.get("real_train_rows", run.get("train_rows")),
            "synthetic_rows": run.get("synthetic_rows", 0), "train_rows": run.get("train_rows"),
            "validation_macro_f1": validation.get("eval_macro_f1", validation.get("macro_f1")),
            "test_accuracy": test.get("accuracy"), "test_macro_precision": test.get("macro_precision"),
            "test_macro_recall": test.get("macro_recall"), "test_macro_f1": test.get("macro_f1"),
            "test_weighted_f1": test.get("weighted_f1") or (test.get("classification_report") or {}).get("weighted avg", {}).get("f1-score"),
            "inference_latency_ms_per_example": test.get("inference_latency_ms_per_example"),
            "training_seconds": run.get("training_seconds"), "run_metadata_path": run.get("run_metadata_path"),
        })
    table = pd.DataFrame(rows, columns=[
        "experiment_id", "status", "model", "arm", "seed", "proportion", "real_train_rows",
        "synthetic_rows", "train_rows", "validation_macro_f1", "test_accuracy",
        "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_weighted_f1",
        "inference_latency_ms_per_example", "training_seconds", "run_metadata_path",
    ])
    csv_path = output_path.with_suffix(".csv")
    table.to_csv(csv_path, index=False)
    completed = sum(str(run.get("status", "")).upper() == "COMPLETED" for run in runs)
    failed = sum(str(run.get("status", "")).upper() == "FAILED" for run in runs)
    if table.empty:
        result_table = "No saved model run records were found. Model metrics are unavailable; no scores are inferred."
    else:
        display = table.drop(columns=["run_metadata_path"]).copy()
        columns = list(display.columns)
        rows_md = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        rows_md.extend("| " + " | ".join("" if pd.isna(value) else str(value).replace("|", "\\|") for value in row) + " |" for row in display.itertuples(index=False, name=None))
        result_table = "\n".join(rows_md)
    text = f"""# Marathi MahaHate research report

## Scope

This report describes only saved run metadata for four-class Marathi MahaHate hate-speech classification. It does not claim evaluation of Gujarati, Assamese, sentiment analysis, named entity recognition, or cross-lingual transfer.

Synthetic generation and A0-A6 training interfaces are implemented. Persisted live provider candidates: {len(candidates)}; accepted by machine checks and pending human review: {accepted_candidates}; rejected: {len(candidates) - accepted_candidates}. This is not a ground-truth evaluation and no augmentation-arm training result is recorded. Fixture smoke output is excluded. A0 means real-only runs; historical records without an explicit arm are reported as A0 for compatibility.

## Run status

- Saved run records: {len(runs)}
- Completed: {completed}
- Failed: {failed}
- Test metrics are included only when the run record contains them. A missing value means unavailable, not zero.

## Recorded results

{result_table}

## Method and limitations

The run script selects checkpoints using validation Macro-F1. Held-out test evaluation is opt-in and must happen after model selection. This report does not independently verify the underlying data or checkpoint. Review the run JSON, split fingerprints, model configuration, and experiment registry before using results in a thesis. No statistical significance is claimed from a single seed.

## Reproducibility

Each row links to its saved run metadata path in the adjacent CSV. If there are no run records, see the implementation audit and status report for missing data, environment, and experiments.
"""
    output_path.write_text(text, encoding="utf-8")
    return output_path, csv_path
