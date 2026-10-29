"""Local inference API for the saved Marathi hate-speech classifiers."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from research_api import router as research_router, training_active

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "results" / "models"
LABELS = ["hate", "offensive", "profane", "not"]
LABEL_TITLES = {
    "hate": "Hate speech",
    "offensive": "Offensive language",
    "profane": "Profane language",
    "not": "Not abusive",
}
LABEL_HELP = {
    "hate": "Abusive language aimed at a group or community.",
    "offensive": "Insulting, dehumanizing, or threatening language aimed at an individual.",
    "profane": "Swear words or profane language without a clear targeted group attack.",
    "not": "No clear abusive or profane content was detected.",
}
LOGGER = logging.getLogger("marathi_nlp.api")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_models: dict[str, tuple[Any, Any]] = {}
_model_lock = threading.Lock()


def _read_run(path: Path) -> dict[str, Any] | None:
    try:
        run = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if str(run.get("status", "")).upper() != "COMPLETED":
        return None
    if run.get("validation_full_split") is False:
        return None
    checkpoint = path.parent / "best"
    if not (checkpoint / "config.json").is_file():
        return None
    if not any((checkpoint / weight).is_file() for weight in ("model.safetensors", "pytorch_model.bin")):
        return None
    validation = run.get("validation_metrics") or {}
    test = run.get("test_metrics") or {}
    return {
        "id": path.parent.name,
        "name": run.get("model", "Classifier"),
        "experiment_arm": run.get("experiment_arm", "A0"),
        "proportion": run.get("effective_proportion", run.get("proportion")),
        "checkpoint": str(checkpoint),
        "train_rows": int(run.get("train_rows") or 0),
        "real_train_rows": int(run.get("real_train_rows", run.get("train_rows") or 0)),
        "synthetic_rows": int(run.get("synthetic_rows") or 0),
        "validation_rows": int(run.get("validation_rows") or 0),
        "validation_macro_f1": validation.get("eval_macro_f1"),
        "test_macro_precision": test.get("macro_precision"),
        "test_macro_recall": test.get("macro_recall"),
        "test_macro_f1": test.get("macro_f1"),
        "test_accuracy": test.get("accuracy"),
        "test_weighted_f1": test.get("weighted_f1") or (test.get("classification_report") or {}).get("weighted avg", {}).get("f1-score"),
        "inference_latency_ms": test.get("inference_latency_ms_per_example"),
        "test_per_class": {
            label: (test.get("classification_report") or {}).get(label, {}).get("f1-score")
            for label in LABELS
        },
        "seed": run.get("seed"),
        "epochs": (validation.get("epoch")),
        "device": run.get("device"),
        "training_seconds": run.get("training_seconds"),
    }


def available_models() -> list[dict[str, Any]]:
    if not MODEL_DIR.exists():
        return []
    records = [r for path in MODEL_DIR.rglob("run.json") if (r := _read_run(path))]
    # Validation Macro-F1 is used for the default choice. Held-out test data
    # is reported but is never used to rank deployment candidates.
    return sorted(
        records,
        key=lambda r: (float(r.get("validation_macro_f1") or 0), r["train_rows"], r["id"]),
        reverse=True,
    )


def _get_model(model_id: str) -> tuple[Any, Any, dict[str, Any]]:
    record = next((m for m in available_models() if m["id"] == model_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="That completed model checkpoint is unavailable.")
    with _model_lock:
        if model_id not in _models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(record["checkpoint"], local_files_only=True)
                classifier = AutoModelForSequenceClassification.from_pretrained(
                    record["checkpoint"], local_files_only=True
                ).to(DEVICE)
                classifier.eval()
                _models[model_id] = (tokenizer, classifier)
            except Exception as exc:
                LOGGER.exception("Could not load checkpoint %s", model_id)
                raise HTTPException(
                    status_code=503,
                    detail=f"Could not load the local {record['name']} checkpoint ({type(exc).__name__}).",
                ) from exc
    tokenizer, classifier = _models[model_id]
    return tokenizer, classifier, record


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    model_id: str | None = None


app = FastAPI(title="Marathi NLP Lab API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(research_router)


@app.get("/api/health")
def health() -> dict[str, Any]:
    models = available_models()
    return {
        "status": "ok" if models else "model_missing",
        "device": DEVICE,
        "models_available": len(models),
        "loaded_model_ids": list(_models),
        "training_active": training_active(),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    records = available_models()
    return {
        "models": [
            {
                **record,
                "is_default": index == 0,
                "loaded": record["id"] in _models,
            }
            for index, record in enumerate(records)
        ],
        "device": DEVICE,
        "labels": [{"id": key, "title": LABEL_TITLES[key], "description": LABEL_HELP[key]} for key in LABELS],
    }


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if training_active():
        raise HTTPException(status_code=409, detail="Inference is paused while the selected experiment uses the GPU.")
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Enter some Marathi text before analyzing it.")
    records = available_models()
    model_id = request.model_id or (records[0]["id"] if records else None)
    if model_id is None:
        raise HTTPException(status_code=503, detail="No completed model checkpoint was found in results/models.")
    tokenizer, classifier, record = _get_model(model_id)
    try:
        batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        with torch.inference_mode():
            logits = classifier(**batch).logits
            probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
    except Exception as exc:
        LOGGER.exception("Inference failed for checkpoint %s", model_id)
        raise HTTPException(status_code=500, detail=f"Inference failed ({type(exc).__name__}).") from exc
    if len(probabilities) != len(LABELS):
        raise HTTPException(status_code=500, detail="The saved model does not have the expected four-class head.")
    winner = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "text": text,
        "predicted_label": LABELS[winner],
        "predicted_title": LABEL_TITLES[LABELS[winner]],
        "description": LABEL_HELP[LABELS[winner]],
        "confidence": probabilities[winner],
        "probabilities": [
            {"label": label, "title": LABEL_TITLES[label], "probability": probabilities[i]}
            for i, label in enumerate(LABELS)
        ],
        "model": {key: record[key] for key in ("id", "name", "train_rows", "validation_macro_f1", "test_macro_f1")},
        "device": DEVICE,
        "max_tokens": 128,
        "truncated": len(tokenizer(text, truncation=True, max_length=128)["input_ids"]) >= 128,
    }
