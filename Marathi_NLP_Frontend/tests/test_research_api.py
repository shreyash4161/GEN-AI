from fastapi.testclient import TestClient
import os
import pandas as pd
from fastapi import HTTPException
import pytest

from api import app
import synthetic_pipeline
import research_api


def test_research_api_exposes_scope_dataset_and_report(monkeypatch):
    client = TestClient(app)
    scope = client.get("/api/scope")
    assert scope.status_code == 200
    assert scope.json()["verified_combinations"][0]["status"] == "IMPLEMENTED_AND_EVALUATED"

    dataset = client.get("/api/dataset/stats")
    assert dataset.status_code == 200
    assert dataset.json()["total_rows"] > 0
    assert "test" in dataset.json()["splits"]

    experiments = client.get("/api/experiments")
    assert experiments.status_code == 200
    assert set(experiments.json()["arms"]) == {f"A{i}" for i in range(7)}


def test_generation_reports_missing_live_provider_configuration(tmp_path, monkeypatch):
    # Isolate this test from the real project .env so it can never invoke Groq.
    monkeypatch.setattr(synthetic_pipeline, "ROOT", tmp_path)
    for name in ("GROQ_API_KEY", "GROQ_GENERATOR_MODEL", "GROQ_VALIDATOR_MODEL"):
        monkeypatch.delenv(name, raising=False)
    response = TestClient(app).post("/api/generate", json={"label": "not"})
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_api_reconciles_windows_csv_pid_with_decimal_suffix(tmp_path, monkeypatch):
    registry = tmp_path / "experiment_registry.csv"
    pd.DataFrame([{
        "experiment_id": "active-run", "status": "RUNNING", "pid": "9134.0",
        "model": "MahaBERT", "experiment_arm": "A0", "output": str(tmp_path / "out"),
    }]).to_csv(registry, index=False)
    monkeypatch.setattr(research_api, "REGISTRY_PATH", registry)
    monkeypatch.setattr(research_api, "_active_process", None)
    monkeypatch.setattr(research_api, "_active_run_id", None)
    monkeypatch.setattr(research_api, "_external_active_pid", None)
    monkeypatch.setattr(os, "kill", lambda pid, signal: None)

    research_api._reconcile_active_process()
    assert research_api._active_run_id == "active-run"
    assert research_api._external_active_pid == 9134
    assert research_api._training_is_active() is True


def test_experiment_start_blocks_when_disk_is_below_checkpoint_requirement(monkeypatch):
    monkeypatch.setattr(research_api, "_reconcile_active_process", lambda: None)
    monkeypatch.setattr(research_api, "_training_is_active", lambda: False)
    monkeypatch.setattr(research_api.shutil, "disk_usage", lambda _: type("Usage", (), {"free": int(1.27 * 1024**3)})())
    request = research_api.ExperimentRequest(model="MahaBERT", resumable_checkpoints=True)

    with pytest.raises(HTTPException) as error:
        research_api.start_experiment(request)
    assert error.value.status_code == 507
    assert "4 GiB" in error.value.detail


def test_dataset_and_synthetic_downloads_are_allowlisted(tmp_path, monkeypatch):
    monkeypatch.setattr(research_api, "ROOT", tmp_path)
    dataset = tmp_path / "data" / "maha_hate" / "hate_train.xlsx"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"local dataset bytes")
    generation_id = "gen_20261029T000000_ab12cd34"
    candidate = tmp_path / "results" / "generated_data" / f"{generation_id}_candidates.jsonl"
    candidate.parent.mkdir(parents=True)
    candidate.write_text('{"text":"candidate"}\n', encoding="utf-8")
    accepted = tmp_path / "data" / "synthetic" / "accepted" / f"{generation_id}_accepted.jsonl"
    accepted.parent.mkdir(parents=True)
    accepted.write_text('{"text":"accepted","human_review_required":true}\n', encoding="utf-8")
    client = TestClient(app)

    dataset_response = client.get("/api/dataset/download/train")
    assert dataset_response.status_code == 200
    assert dataset_response.content == b"local dataset bytes"
    assert "hate_train.xlsx" in dataset_response.headers["content-disposition"]
    assert client.get("/api/dataset/download/other").status_code == 422

    candidate_response = client.get(f"/api/generated/{generation_id}/download?kind=candidates")
    accepted_response = client.get(f"/api/generated/{generation_id}/download?kind=accepted")
    assert candidate_response.status_code == accepted_response.status_code == 200
    assert b'"candidate"' in candidate_response.content
    assert b'"human_review_required":true' in accepted_response.content
    assert client.get("/api/generated/../../outside/download?kind=candidates").status_code in {404, 422}
