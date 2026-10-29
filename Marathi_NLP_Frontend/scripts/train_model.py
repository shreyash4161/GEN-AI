"""Train one configured classifier. The held-out test split is opt-in and loaded last."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import hashlib
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_core import ExperimentRegistry, LABELS, load_dataset, stratified_subset
from experiment_core import select_synthetic_examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("MahaBERT", "MuRIL", "XLM-RoBERTa", "IndicBERT v2"), required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "maha_hate")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "models")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proportion", type=float, default=1.0)
    parser.add_argument("--experiment-arm", choices=("A0", "A1", "A2", "A3", "A4", "A5", "A6"), default="A0")
    parser.add_argument("--synthetic-jsonl", type=Path, default=None, help="Candidate JSONL saved by the synthetic lab")
    parser.add_argument("--synthetic-budget", type=int, default=0, help="Maximum synthetic rows to add for A1-A6")
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None, help="Use a small stratified sample for smoke runs")
    parser.add_argument("--max-validation-rows", type=int, default=None, help="Use a validation-only subset for resource-limited smoke runs")
    parser.add_argument("--resumable-checkpoints", action="store_true", help="Save optimizer/scheduler state too; requires substantially more disk space")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate test split once after validation checkpoint selection")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 8:
        parser.error("epochs, batch-size, and max-length must be positive reasonable values")
    if args.synthetic_budget < 0:
        parser.error("synthetic-budget cannot be negative")
    if args.experiment_arm != "A0" and (args.synthetic_jsonl is None or args.synthetic_budget < 1):
        parser.error("A1-A6 require --synthetic-jsonl and a positive --synthetic-budget")
    if args.experiment_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", args.experiment_id):
        parser.error("experiment-id may contain only letters, digits, underscore, and hyphen")
    if args.resume_from_checkpoint and not args.resumable_checkpoints:
        parser.error("resume-from-checkpoint requires --resumable-checkpoints")
    disk_path = args.output_dir if args.output_dir.exists() else args.output_dir.parent
    required_gib = 4.0 if args.resumable_checkpoints else 2.0
    free_gib = shutil.disk_usage(disk_path).free / 1024**3
    if free_gib < required_gib:
        parser.error(
            f"Only {free_gib:.2f} GiB free; this run requires at least {required_gib:.0f} GiB "
            f"with resumable checkpoints {'enabled' if args.resumable_checkpoints else 'disabled'}. "
            "Free disk space or disable optimizer checkpoints before training."
        )
    config = json.loads((ROOT / "configs" / "models.json").read_text(encoding="utf-8"))
    checkpoint = config["models"][args.model]
    run_id = args.experiment_id or f"{args.model.lower().replace(' ', '-')}_{args.experiment_arm.lower()}_{int(time.time())}_seed{args.seed}"
    run_dir = args.output_dir / run_id
    registry = ExperimentRegistry(ROOT / "results" / "experiment_registry.csv")
    registry.update(run_id, "RUNNING", model=args.model, checkpoint=checkpoint, seed=args.seed,
                    experiment_arm=args.experiment_arm, proportion=args.proportion,
                    epochs=args.epochs, batch_size=args.batch_size, pid=os.getpid(),
                    output=str(run_dir.resolve()), synthetic_file=str(args.synthetic_jsonl.resolve()) if args.synthetic_jsonl else "",
                    synthetic_budget=args.synthetic_budget, resumable_checkpoints=args.resumable_checkpoints)
    start = time.perf_counter()
    try:
        import torch
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                                  Trainer, TrainingArguments, set_seed)

        set_seed(args.seed)
        train, train_meta = load_dataset(args.data_dir / "hate_train.xlsx")
        valid, valid_meta = load_dataset(args.data_dir / "hate_valid.xlsx")
        validation_texts = set(valid.text)
        overlap_mask = train.text.isin(validation_texts)
        train_validation_overlap_removed = int(overlap_mask.sum())
        train = train.loc[~overlap_mask].reset_index(drop=True)
        eligible_train_rows = len(train)
        if train_validation_overlap_removed:
            print(f"Removed {train_validation_overlap_removed} exact train/validation duplicate texts from training.")
        train = stratified_subset(train, args.proportion, args.seed)
        if args.max_train_rows:
            if args.max_train_rows < len(LABELS):
                raise ValueError("max-train-rows must allow at least one row per class")
            train = stratified_subset(train, min(1.0, args.max_train_rows / len(train)), args.seed)
        if args.max_validation_rows:
            if args.max_validation_rows < len(LABELS):
                raise ValueError("max-validation-rows must allow at least one row per class")
            valid = stratified_subset(valid, min(1.0, args.max_validation_rows / len(valid)), args.seed)
        real_train_rows = len(train)
        synthetic_rows_available = 0
        selected_synthetic_rows: list[dict[str, object]] = []
        if args.experiment_arm != "A0":
            if not args.synthetic_jsonl or not args.synthetic_jsonl.is_file():
                raise FileNotFoundError(f"Synthetic JSONL does not exist: {args.synthetic_jsonl}")
            candidate_records = []
            for line_number, line in enumerate(args.synthetic_jsonl.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    candidate_records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid synthetic JSONL at line {line_number}: {exc}") from exc
            synthetic_rows_available = len(candidate_records)
            selected_synthetic_rows = select_synthetic_examples(
                candidate_records, args.experiment_arm, args.synthetic_budget, args.seed,
                real_training_texts=[*train.text.tolist(), *valid.text.tolist()],
            )
            if not selected_synthetic_rows:
                raise ValueError(f"No eligible synthetic rows remain for {args.experiment_arm}; check acceptance, targeting, and budget.")
            synthetic_frame = pd.DataFrame(selected_synthetic_rows)[["text", "label"]]
            train = pd.concat([train, synthetic_frame], ignore_index=True)
            if set(train.label) != set(LABELS):
                raise ValueError("Combined real/synthetic training data lost one or more required classes")
            print(f"Added {len(synthetic_frame)} synthetic rows for {args.experiment_arm} from {synthetic_rows_available} candidate records.")
        if set(train.label) != set(LABELS) or set(valid.label) != set(LABELS):
            raise ValueError("Train and validation must both contain all four labels")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=4, id2label={i: label for i, label in enumerate(LABELS)},
            label2id={label: i for i, label in enumerate(LABELS)},
        )

        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, frame: pd.DataFrame):
                self.texts = frame.text.tolist()
                self.labels = [LABELS.index(label) for label in frame.label]

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, index):
                item = tokenizer(self.texts[index], truncation=True, max_length=args.max_length)
                item["labels"] = self.labels[index]
                return item

        train_set, valid_set = TextDataset(train), TextDataset(valid)

        def metrics(eval_pred):
            logits, y = eval_pred
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = np.argmax(logits, axis=1)
            p, r, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(4)), average="macro", zero_division=0)
            return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f), "macro_precision": float(p), "macro_recall": float(r)}

        training_args = TrainingArguments(
            output_dir=str(run_dir / "training"), num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=min(args.batch_size, 2),
            learning_rate=2e-5, weight_decay=.01, eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="eval_macro_f1", greater_is_better=True,
            save_total_limit=1, save_only_model=not args.resumable_checkpoints,
            logging_strategy="epoch", report_to="none", seed=args.seed,
            fp16=torch.cuda.is_available(), dataloader_num_workers=0, eval_accumulation_steps=1,
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=valid_set,
                          processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
        if args.resume_from_checkpoint:
            resume_path = args.resume_from_checkpoint.resolve()
            if not resume_path.is_relative_to((run_dir / "training").resolve()) or not (resume_path / "optimizer.pt").is_file():
                raise ValueError("Resume checkpoint must be an optimizer checkpoint inside this run's training directory")
            trainer.train(resume_from_checkpoint=str(resume_path))
        else:
            trainer.train()
        run_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(run_dir / "best"))
        tokenizer.save_pretrained(run_dir / "best")
        validation = trainer.evaluate(valid_set)
        result = {
            "experiment_id": run_id, "status": "COMPLETED", "model": args.model,
            "experiment_arm": args.experiment_arm,
            "checkpoint": checkpoint, "seed": args.seed, "proportion": args.proportion,
            "effective_proportion": real_train_rows / eligible_train_rows,
            "eligible_train_rows": eligible_train_rows,
            "requested_proportion": args.proportion,
            "train_rows": len(train), "real_train_rows": real_train_rows,
            "synthetic_rows": len(selected_synthetic_rows),
            "synthetic_rows_available": synthetic_rows_available,
            "synthetic_budget": args.synthetic_budget,
            "synthetic_file": str(args.synthetic_jsonl.resolve()) if args.synthetic_jsonl else None,
            "validation_rows": len(valid),
            "validation_subset_rows": len(valid), "validation_full_split": args.max_validation_rows is None,
            "optimizer_state_saved": args.resumable_checkpoints,
            "resumable_checkpoints": args.resumable_checkpoints,
            "train_validation_overlap_removed": train_validation_overlap_removed,
            "train_fingerprint": train_meta["sha256"], "validation_fingerprint": valid_meta["sha256"],
            "effective_train_fingerprint": hashlib.sha256("\n".join(train.text + "\t" + train.label).encode("utf-8")).hexdigest(),
            "validation_metrics": validation, "test_metrics": None,
            "training_seconds": time.perf_counter() - start,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "software": {"torch": torch.__version__}, "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if args.evaluate_test:
            test, test_meta = load_dataset(args.data_dir / "hate_test.xlsx")
            prediction = trainer.predict(TextDataset(test))
            pred = np.argmax(prediction.predictions, axis=1)
            y = np.array([LABELS.index(label) for label in test.label])
            p, r, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(4)), average="macro", zero_division=0)
            result["test_metrics"] = {
                "accuracy": float(accuracy_score(y, pred)), "macro_precision": float(p),
                "macro_recall": float(r), "macro_f1": float(f),
                "weighted_f1": float(classification_report(y, pred, labels=list(range(4)), target_names=LABELS, output_dict=True, zero_division=0)["weighted avg"]["f1-score"]),
                "inference_latency_ms_per_example": float(prediction.metrics.get("test_runtime", 0)) * 1000 / max(1, len(test)),
                "classification_report": classification_report(y, pred, labels=list(range(4)), target_names=LABELS, output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(y, pred, labels=list(range(4))).tolist(),
                "dataset_fingerprint": test_meta["sha256"],
            }
        (run_dir / "run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        registry.update(run_id, "COMPLETED", model=args.model, checkpoint=checkpoint, seed=args.seed,
                        experiment_arm=args.experiment_arm, synthetic_rows=len(selected_synthetic_rows),
                        proportion=args.proportion, epochs=args.epochs, batch_size=args.batch_size,
                        pid=os.getpid(), output=str(run_dir.resolve()),
                        synthetic_file=str(args.synthetic_jsonl.resolve()) if args.synthetic_jsonl else "",
                        synthetic_budget=args.synthetic_budget, resumable_checkpoints=args.resumable_checkpoints,
                        validation_macro_f1=validation.get("eval_macro_f1"))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        registry.update(run_id, "FAILED", error=message, model=args.model, checkpoint=checkpoint, seed=args.seed,
                        experiment_arm=args.experiment_arm, proportion=args.proportion,
                        epochs=args.epochs, batch_size=args.batch_size, pid=os.getpid(), output=str(run_dir.resolve()),
                        synthetic_file=str(args.synthetic_jsonl.resolve()) if args.synthetic_jsonl else "",
                        synthetic_budget=args.synthetic_budget, resumable_checkpoints=args.resumable_checkpoints)
        print(f"Training failed; status recorded as FAILED. {message}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
