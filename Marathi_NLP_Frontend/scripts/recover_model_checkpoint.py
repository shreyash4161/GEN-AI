"""Recover/evaluate a model-only checkpoint after optimizer-state save failure.

This does not continue training. It evaluates the last saved model on the full
validation split first, then the held-out test split, and records the original
checkpoint-save error in run metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_core import ExperimentRegistry, LABELS, load_dataset, stratified_subset


def evaluate(model, tokenizer, frame: pd.DataFrame, batch_size: int, device: str) -> tuple[np.ndarray, float]:
    import torch
    predictions: list[int] = []
    started = time.perf_counter()
    model.eval()
    for offset in range(0, len(frame), batch_size):
        texts = frame.text.iloc[offset:offset + batch_size].tolist()
        encoded = tokenizer(texts, return_tensors="pt", truncation=True, max_length=128, padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
        predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
    return np.asarray(predictions), time.perf_counter() - started


def metrics(frame: pd.DataFrame, predictions: np.ndarray, elapsed: float) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    labels = np.asarray([LABELS.index(value) for value in frame.label])
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=list(range(4)), average="macro", zero_division=0,
    )
    report = classification_report(
        labels, predictions, labels=list(range(4)), target_names=LABELS,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision), "macro_recall": float(recall),
        "macro_f1": float(macro_f1), "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "inference_latency_ms_per_example": elapsed * 1000 / max(1, len(frame)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(4))).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    registry_path = ROOT / "results" / "experiment_registry.csv"
    registry = ExperimentRegistry(registry_path)
    rows = registry.read()
    match = rows.loc[rows.experiment_id.astype(str).eq(args.experiment_id)]
    if match.empty:
        parser.error("Experiment ID is not present in the saved registry")
    record = match.iloc[0].to_dict()
    if str(record.get("status", "")).upper() != "FAILED":
        parser.error("Checkpoint recovery is allowed only for a failed run")
    if str(record.get("experiment_arm", "A0")).upper() != "A0":
        parser.error("Recovery currently supports real-only A0 runs only")
    run_dir = Path(str(record.get("output", ""))).resolve()
    if not run_dir.is_relative_to((ROOT / "results" / "models").resolve()):
        parser.error("Saved output directory is outside results/models")
    training_dir = run_dir / "training"
    checkpoints = [
        path for path in training_dir.glob("checkpoint-*")
        if (path / "config.json").is_file()
        and any((path / weight).is_file() for weight in ("model.safetensors", "pytorch_model.bin"))
    ]
    if not checkpoints:
        parser.error("No complete model-weight checkpoint was found")
    checkpoint = max(checkpoints, key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    best_dir = run_dir / "best"
    reuse_checkpoint_link = best_dir.exists() and best_dir.resolve() == checkpoint.resolve()
    if best_dir.exists() and not reuse_checkpoint_link and (
        (best_dir / "run.json").exists()
        or any((best_dir / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
    ):
        parser.error("A best checkpoint already exists; refusing to overwrite it")
    free_bytes = __import__("shutil").disk_usage(run_dir).free
    required_bytes = 100 * 1024**2 if reuse_checkpoint_link else 1200 * 1024**2
    if free_bytes < required_bytes:
        parser.error(f"Only {free_bytes / 1024**3:.2f} GiB free; insufficient space for checkpoint recovery")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    model_name = str(record["model"])
    config = json.loads((ROOT / "configs" / "models.json").read_text(encoding="utf-8"))
    base_model = config["models"][model_name]
    seed = int(float(record.get("seed", 42)))
    proportion = float(record.get("proportion", 1.0))
    batch_size = max(1, min(4, int(float(record.get("batch_size", 2)))))
    train, train_meta = load_dataset(ROOT / "data" / "maha_hate" / "hate_train.xlsx")
    valid, valid_meta = load_dataset(ROOT / "data" / "maha_hate" / "hate_valid.xlsx")
    validation_texts = set(valid.text)
    overlap = train.text.isin(validation_texts)
    overlap_removed = int(overlap.sum())
    train = train.loc[~overlap].reset_index(drop=True)
    eligible_rows = len(train)
    train = stratified_subset(train, proportion, seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device)

    # The checkpoint directory contains the only saved epoch in this run. The
    # original training log has its full-validation evaluation before the
    # optimizer-save error; re-evaluate it here before touching test data.
    validation_predictions, validation_seconds = evaluate(model, tokenizer, valid, batch_size, device)
    validation_metric = metrics(valid, validation_predictions, validation_seconds)
    validation_metric = {f"eval_{key}": value for key, value in validation_metric.items()}
    if not np.isfinite(validation_metric["eval_macro_f1"]):
        raise RuntimeError("Validation Macro-F1 is not finite; refusing to finalize recovery")

    test, test_meta = load_dataset(ROOT / "data" / "maha_hate" / "hate_test.xlsx")
    test_predictions, test_seconds = evaluate(model, tokenizer, test, batch_size, device)
    test_metric = metrics(test, test_predictions, test_seconds)

    if not reuse_checkpoint_link:
        best_dir.mkdir(parents=True, exist_ok=True)
        # Copy already-saved weights instead of serializing the large model.
        for filename in (
            "config.json", "model.safetensors", "pytorch_model.bin",
            "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
            "vocab.txt", "sentencepiece.bpe.model", "spiece.model",
        ):
            source = checkpoint / filename
            if source.is_file():
                shutil.copy2(source, best_dir / filename)
        if not any((best_dir / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            raise RuntimeError("Could not copy complete model weights to the recovered best directory")
    effective_train_fingerprint = hashlib.sha256(
        "\n".join(train.text + "\t" + train.label).encode("utf-8")
    ).hexdigest()
    result = {
        "experiment_id": args.experiment_id, "status": "COMPLETED",
        "model": model_name, "experiment_arm": "A0", "checkpoint": base_model,
        "seed": seed, "proportion": proportion, "requested_proportion": proportion,
        "effective_proportion": len(train) / max(1, eligible_rows),
        "eligible_train_rows": eligible_rows, "train_rows": len(train), "real_train_rows": len(train),
        "synthetic_rows": 0, "validation_rows": len(valid), "validation_subset_rows": len(valid),
        "validation_full_split": True, "optimizer_state_saved": False,
        "resumable_checkpoints": False, "train_validation_overlap_removed": overlap_removed,
        "train_fingerprint": train_meta["sha256"], "validation_fingerprint": valid_meta["sha256"],
        "effective_train_fingerprint": effective_train_fingerprint,
        "validation_metrics": validation_metric, "test_metrics": {**test_metric, "dataset_fingerprint": test_meta["sha256"]},
        "training_seconds": None, "device": device, "software": {"torch": torch.__version__},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_recovery": {
            "recovered_from": str(checkpoint.relative_to(ROOT)), "checkpoint_step": int(checkpoint.name.rsplit("-", 1)[-1]),
            "status": "RECOVERED_MODEL_ONLY_CHECKPOINT",
            "original_training_error": str(record.get("error", "")),
            "note": "Training epoch and validation evaluation completed; optimizer-state serialization failed. This model-only checkpoint was re-evaluated on full validation, then held-out test. Training time was unavailable.",
        },
    }
    (run_dir / "run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    registry.update(
        args.experiment_id, "COMPLETED", model=model_name, checkpoint=base_model, seed=seed,
        experiment_arm="A0", proportion=proportion, epochs=record.get("epochs", 1),
        batch_size=batch_size, pid="", output=str(run_dir.resolve()), resumable_checkpoints=False,
        checkpoint_recovery_error=str(record.get("error", "")),
        validation_macro_f1=validation_metric["eval_macro_f1"],
    )
    print(json.dumps({
        "status": result["status"], "experiment_id": args.experiment_id,
        "validation_macro_f1": validation_metric["eval_macro_f1"],
        "test_macro_f1": test_metric["macro_f1"], "test_accuracy": test_metric["accuracy"],
        "checkpoint": str(best_dir), "training_seconds": None,
        "note": result["checkpoint_recovery"]["note"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
