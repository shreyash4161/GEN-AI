"""Small, dependency-light research utilities shared by scripts and dashboard.

These helpers deliberately do not claim model-based label validation when no
independent validator is available. The score is a transparent heuristic, not
a substitute for human review.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

LABELS = ("hate", "offensive", "profane", "not")
LABEL_MAP = {
    "hate": "hate", "offn": "offensive", "offensive": "offensive",
    "prfn": "profane", "profane": "profane", "not": "not",
}
EXPERIMENTS = {
    "A0": "real_only", "A1": "random_synthetic",
    "A2": "class_balanced", "A3": "linguistic_control",
    "A4": "sdqs_filtered", "A5": "sdqs_difficulty",
    "A6": "adaptive_quality_aware",
}


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip()


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def load_dataset(path: str | Path, text_col: str | None = None,
                 label_col: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and normalize a labeled dataset; fail explicitly on unknown labels."""
    path = Path(path)
    frame = _read_frame(path)
    columns = {str(c).strip().lower(): c for c in frame.columns}
    text_col = text_col or next((columns[x] for x in ("text", "tweet", "sentence", "content", "post") if x in columns), None)
    label_col = label_col or next((columns[x] for x in ("label", "class", "category") if x in columns), None)
    if text_col not in frame.columns or label_col not in frame.columns:
        raise ValueError(f"Could not identify text/label columns in {list(frame.columns)}")
    out = frame[[text_col, label_col]].copy()
    out.columns = ["text", "label_raw"]
    input_rows = len(out)
    out["text"] = out["text"].map(normalize_text)
    out["label"] = out["label_raw"].astype("string").str.strip().str.lower().map(LABEL_MAP)
    missing_or_empty = int((out.text.eq("") | out.label.isna()).sum())
    out = out.loc[out.text.ne("") & out.label.notna(), ["text", "label"]].copy()
    conflicting = out.groupby("text").label.nunique().gt(1)
    conflicting_texts = int(conflicting.sum())
    conflicting_values = set(conflicting[conflicting].index)
    conflict_rows_removed = int(out.text.isin(conflicting_values).sum())
    out = out.loc[~out.text.isin(conflicting_values)].copy()
    duplicate_rows = int(out.duplicated("text", keep="first").sum())
    out = out.drop_duplicates("text", keep="first").reset_index(drop=True)
    metadata = {
        "path": str(path), "input_rows": input_rows, "rows": len(out),
        "dropped_invalid_rows": missing_or_empty, "duplicate_rows_removed": duplicate_rows,
        "conflicting_texts": conflicting_texts, "conflicting_rows_removed": conflict_rows_removed,
        "label_counts": out.label.value_counts().reindex(LABELS, fill_value=0).to_dict(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return out, metadata


def stratified_subset(frame: pd.DataFrame, proportion: float, seed: int = 42) -> pd.DataFrame:
    if not 0 < proportion <= 1:
        raise ValueError("proportion must be in (0, 1]")
    if proportion == 1:
        return frame.reset_index(drop=True).copy()
    # sklearn's stratification fails when a class has fewer than two examples.
    counts = frame.label.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Stratified sampling requires at least two examples per present class")
    from sklearn.model_selection import train_test_split
    n = max(len(counts), round(len(frame) * proportion))
    n = min(n, len(frame) - len(counts))
    selected, _ = train_test_split(frame, train_size=n, random_state=seed, stratify=frame.label)
    return selected.reset_index(drop=True)


def assert_no_split_leakage(*splits: pd.DataFrame) -> None:
    seen: set[str] = set()
    for index, frame in enumerate(splits):
        texts = set(frame.text.map(normalize_text))
        overlap = seen.intersection(texts)
        if overlap:
            raise ValueError(f"Exact text leakage detected in split {index}: {len(overlap)} overlapping texts")
        seen.update(texts)


def difficulty_category(confidence: float | None, margin: float | None = None) -> str:
    """Assign difficulty from a classifier's confidence/margin, when available."""
    if confidence is None or not math.isfinite(float(confidence)):
        return "unscored"
    c = float(confidence)
    m = float(margin) if margin is not None and math.isfinite(float(margin)) else c
    if c < .40 or m < .10:
        return "hard"
    if c < .65 or m < .25:
        return "medium"
    return "easy"


@dataclass(frozen=True)
class SDQSConfig:
    weights: Mapping[str, float]
    threshold: float = 0.70
    max_duplicate_similarity: float = 0.90

    def validate(self) -> None:
        if not self.weights or any(not math.isfinite(float(w)) or float(w) < 0 for w in self.weights.values()):
            raise ValueError("SDQS weights must be finite, nonnegative, and nonempty")
        if not math.isclose(sum(map(float, self.weights.values())), 1.0, abs_tol=1e-8):
            raise ValueError("SDQS weights must sum to 1")
        if not 0 <= self.threshold <= 1 or not 0 <= self.max_duplicate_similarity <= 1:
            raise ValueError("SDQS thresholds must be in [0, 1]")


DEFAULT_SDQS = SDQSConfig({"linguistic": .25, "semantic": .20, "label": .30, "novelty": .15, "relevance": .10})


def score_candidate(text: str, requested_label: str, train_texts: Iterable[str],
                    config: SDQSConfig = DEFAULT_SDQS, *, validator: Callable[[str], tuple[str, float]] | None = None,
                    semantic_score: float | None = None, duplicate_similarity: float | None = None,
                    difficulty: str = "unscored", metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Score a candidate; acceptance requires an independent validator result."""
    config.validate()
    text = normalize_text(text)
    if requested_label not in LABELS:
        raise ValueError(f"Unknown requested label: {requested_label}")
    if not text:
        raise ValueError("Candidate text is empty")
    tokens = text.split()
    # Transparent surface-level proxy. No claim of grammatical correctness.
    linguistic = min(1.0, len(tokens) / 8) * (.7 + .3 * min(1.0, len(set(tokens)) / max(1, len(tokens))))
    reference = [normalize_text(x) for x in train_texts]
    exact_duplicate = text.casefold() in {x.casefold() for x in reference}
    duplicate_similarity = 1.0 if exact_duplicate else (float(duplicate_similarity) if duplicate_similarity is not None else 0.0)
    if not 0 <= duplicate_similarity <= 1:
        raise ValueError("duplicate_similarity must be in [0, 1]")
    if validator is None:
        validated_label, label_score = None, 0.0
        validator_type = "unavailable"
    else:
        validated_label, label_score = validator(text)
        validator_type = "independent_model_estimate"
        if validated_label not in LABELS or not 0 <= float(label_score) <= 1:
            raise ValueError("Validator must return a canonical label and score in [0, 1]")
    components = {
        "linguistic": float(linguistic),
        "semantic": float(semantic_score) if semantic_score is not None else 0.0,
        "label": float(label_score) if validated_label == requested_label else 0.0,
        "novelty": 0.0 if exact_duplicate else 1.0 - duplicate_similarity,
        "relevance": 1.0 if len(tokens) >= 2 else 0.0,
    }
    for key in config.weights:
        if key not in components or not 0 <= components[key] <= 1:
            raise ValueError(f"Missing or invalid SDQS component: {key}")
    total = sum(float(weight) * components[key] for key, weight in config.weights.items())
    reasons = []
    if validator is None: reasons.append("independent label validator unavailable")
    elif validated_label != requested_label: reasons.append("validator label disagreement")
    if exact_duplicate: reasons.append("exact duplicate")
    if duplicate_similarity >= config.max_duplicate_similarity: reasons.append("near duplicate threshold exceeded")
    if total < config.threshold: reasons.append("SDQS below threshold")
    return {
        "text": text, "requested_label": requested_label, "validated_label": validated_label,
        "validator_type": validator_type, "components": components, "sdqs": total,
        "duplicate_similarity": duplicate_similarity, "difficulty": difficulty,
        "accepted": not reasons, "rejection_reasons": reasons,
        "generation_metadata": dict(metadata or {}),
    }


def write_record(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one JSONL record without overwriting prior candidate history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**record, "recorded_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")


class ExperimentRegistry:
    STATES = {"NOT_STARTED", "RUNNING", "COMPLETED", "FAILED", "NOT_RUN"}

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame(columns=["experiment_id", "status", "error"])

    def update(self, experiment_id: str, status: str, error: str | None = None, **metadata: Any) -> None:
        status = status.upper()
        if status not in self.STATES:
            raise ValueError(f"Invalid experiment status: {status}")
        frame = self.read()
        row = {"experiment_id": experiment_id, "status": status, "error": error or "", **metadata}
        frame = frame.loc[frame.experiment_id.ne(experiment_id)]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.path, index=False)
