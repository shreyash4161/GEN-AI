import json

import pandas as pd
import pytest

from research_core import (
    DEFAULT_SDQS, ExperimentRegistry, SDQSConfig, assert_no_split_leakage,
    difficulty_category, load_dataset, score_candidate, stratified_subset,
)
from reporting import generate_report


def test_dataset_loader_maps_labels_and_reports_deduplication(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"text": [" मराठी  मजकूर ", "मराठी मजकूर", None], "label": ["HATE", "HATE", "OFFN"]}).to_csv(path, index=False)
    frame, meta = load_dataset(path)
    assert frame.to_dict("records") == [{"text": "मराठी मजकूर", "label": "hate"}]
    assert meta["duplicate_rows_removed"] == 1
    assert meta["dropped_invalid_rows"] == 1


def test_unknown_label_is_not_silently_mapped(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"text": ["मजकूर"], "label": ["maybe"]}).to_csv(path, index=False)
    frame, _ = load_dataset(path)
    assert frame.empty


def test_conflicting_duplicate_labels_are_dropped_and_reported(tmp_path):
    path = tmp_path / "conflict.csv"
    pd.DataFrame({"text": ["एकच वाक्य", "एकच वाक्य", "दुसरे वाक्य"], "label": ["HATE", "NOT", "NOT"]}).to_csv(path, index=False)
    frame, meta = load_dataset(path)
    assert frame.to_dict("records") == [{"text": "दुसरे वाक्य", "label": "not"}]
    assert meta["conflicting_texts"] == 1
    assert meta["conflicting_rows_removed"] == 2


def test_stratified_subset_and_leakage():
    frame = pd.DataFrame({"text": [f"{label} {i}" for label in ("hate", "not") for i in range(10)], "label": [label for label in ("hate", "not") for _ in range(10)]})
    subset = stratified_subset(frame, .5, 7)
    assert len(subset) == 10
    assert subset.label.value_counts().to_dict() == {"hate": 5, "not": 5}
    with pytest.raises(ValueError, match="leakage"):
        assert_no_split_leakage(subset, subset)


def test_sdqs_requires_validator_and_independent_match_for_acceptance():
    record = score_candidate("हा एक वैध मराठी नमुना आहे", "not", [], DEFAULT_SDQS)
    assert record["accepted"] is False
    assert record["validated_label"] is None
    passed = score_candidate("हा एक वैध मराठी नमुना आहे", "not", [], SDQSConfig({"linguistic": .25, "semantic": .20, "label": .30, "novelty": .15, "relevance": .10}, threshold=.5), validator=lambda _: ("not", .95), semantic_score=.9)
    assert passed["accepted"] is True
    rejected = score_candidate("हा एक वैध मराठी नमुना आहे", "not", [], validator=lambda _: ("hate", .99), semantic_score=1)
    assert not rejected["accepted"]
    assert "validator label disagreement" in rejected["rejection_reasons"]


def test_weight_validation_and_duplicate_rejection():
    with pytest.raises(ValueError, match="sum to 1"):
        SDQSConfig({"a": .2, "b": .2}).validate()
    duplicate = score_candidate("समान मराठी वाक्य", "not", ["समान मराठी वाक्य"], validator=lambda _: ("not", 1), semantic_score=1)
    assert not duplicate["accepted"]
    assert "exact duplicate" in duplicate["rejection_reasons"]


@pytest.mark.parametrize("confidence,expected", [(None, "unscored"), (.2, "hard"), (.5, "medium"), (.9, "easy")])
def test_difficulty(confidence, expected):
    assert difficulty_category(confidence) == expected


def test_registry_persists_status_and_error(tmp_path):
    path = tmp_path / "results" / "registry.csv"
    registry = ExperimentRegistry(path)
    registry.update("run-1", "FAILED", error="OOM", model="MuRIL")
    row = registry.read().iloc[0]
    assert row.status == "FAILED" and row.error == "OOM" and row.model == "MuRIL"
    with pytest.raises(ValueError):
        registry.update("run-1", "SUCCESS")


def test_report_generator_preserves_missing_scores(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "one" / "run.json"
    run.parent.mkdir(parents=True)
    run.write_text(json.dumps({"experiment_id": "one", "status": "COMPLETED", "model": "MuRIL", "validation_metrics": {"eval_macro_f1": .5}, "test_metrics": None}), encoding="utf-8")
    md, csv = generate_report(runs, tmp_path / "report.md")
    data = pd.read_csv(csv)
    assert "Completed: 1" in md.read_text(encoding="utf-8")
    assert pd.isna(data.loc[0, "test_macro_f1"])


def test_report_generator_empty_results_does_not_create_scores(tmp_path):
    md, csv = generate_report(tmp_path / "missing", tmp_path / "report.md")
    assert "metrics are unavailable" in md.read_text(encoding="utf-8")
    assert pd.read_csv(csv).empty


def test_report_includes_failed_attempts_from_registry(tmp_path):
    runs = tmp_path / "results" / "models"
    runs.mkdir(parents=True)
    pd.DataFrame([{"experiment_id": "interrupted", "status": "FAILED", "error": "OOM", "model": "MuRIL"}]).to_csv(tmp_path / "results" / "experiment_registry.csv", index=False)
    md, csv = generate_report(runs, tmp_path / "report.md")
    rows = pd.read_csv(csv)
    assert rows.loc[0, "status"] == "FAILED"
    assert "Failed: 1" in md.read_text(encoding="utf-8")


def test_report_counts_persisted_live_generation_separately_from_fixtures(tmp_path):
    runs = tmp_path / "results" / "models"
    generated = tmp_path / "results" / "generated_data"
    runs.mkdir(parents=True)
    generated.mkdir(parents=True)
    records = [
        {"accepted": True, "generation_metadata": {"provider": "openai_compatible"}},
        {"accepted": True, "generation_metadata": {"provider": "fixture"}},
    ]
    (generated / "run_candidates.jsonl").write_text("\n".join(json.dumps(x) for x in records), encoding="utf-8")
    md, _ = generate_report(runs, tmp_path / "report.md")
    content = md.read_text(encoding="utf-8")
    assert "Persisted live provider candidates: 1" in content
    assert "pending human review: 1" in content
