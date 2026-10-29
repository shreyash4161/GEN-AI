"""Deterministic A0-A6 synthetic-row selection for one controlled training run."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable

from research_core import EXPERIMENTS, LABELS, normalize_text


def adaptive_target_from_training_pool(frame: Any, predict: Any, *, per_class: int = 16, seed: int = 42) -> dict[str, Any]:
    """Choose the lowest-recall class from a deterministic training-only sample."""
    if per_class < 1:
        raise ValueError("per_class must be positive")
    rng = random.Random(seed)
    evidence: dict[str, dict[str, int]] = {}
    for label in LABELS:
        group = frame.loc[frame.label.eq(label), ["text", "label"]]
        rows = group.to_dict("records")
        if not rows:
            continue
        chosen = rng.sample(rows, min(per_class, len(rows)))
        correct = 0
        for row in chosen:
            result = predict(row["text"])
            correct += str(result.get("label", "")).lower() == label
        evidence[label] = {"correct": correct, "count": len(chosen)}
    if not evidence:
        raise ValueError("No labeled training examples available for adaptive targeting")
    recalls = {label: counts["correct"] / counts["count"] for label, counts in evidence.items()}
    target = min(LABELS, key=lambda label: (recalls.get(label, 1.0), LABELS.index(label)))
    return {"target_label": target, "per_class_recall": recalls, "per_class_evidence": evidence,
            "method": "lowest_recall_on_deterministic_training_pool_sample", "seed": seed,
            "per_class_sample_limit": per_class, "split_used": "train_only"}


def select_synthetic_examples(
    records: Iterable[dict[str, Any]],
    arm: str,
    budget: int,
    seed: int,
    *,
    real_training_texts: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Select training rows for an arm; no validation/test content is accepted here."""
    arm = arm.upper()
    if arm not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment arm: {arm}")
    if budget < 0:
        raise ValueError("Synthetic budget cannot be negative")
    if arm == "A0" or budget == 0:
        return []

    real = {normalize_text(text).casefold() for text in real_training_texts if normalize_text(text)}
    eligible: list[dict[str, Any]] = []
    seen: set[str] = set(real)
    for source in records:
        text = normalize_text(source.get("text", ""))
        label = source.get("label") or source.get("requested_label")
        if not text or label not in LABELS:
            continue
        key = text.casefold()
        if key in seen:
            continue
        if arm in {"A4", "A5", "A6"} and not source.get("accepted", False):
            continue
        if arm == "A3" and not (source.get("generation_metadata") or {}).get("controlled_style"):
            continue
        if arm == "A5" and source.get("difficulty") not in {"easy", "medium", "hard", "adversarial"}:
            continue
        if arm == "A6" and not (source.get("generation_metadata") or {}).get("adaptive_target"):
            continue
        seen.add(key)
        eligible.append({**source, "text": text, "label": label})

    rng = random.Random(seed)
    if arm == "A2":
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eligible:
            by_label[row["label"]].append(row)
        for group in by_label.values():
            rng.shuffle(group)
        selected: list[dict[str, Any]] = []
        while len(selected) < budget:
            progressed = False
            for label in LABELS:
                if by_label[label] and len(selected) < budget:
                    selected.append(by_label[label].pop())
                    progressed = True
            if not progressed:
                break
        return selected

    rng.shuffle(eligible)
    return eligible[:budget]
