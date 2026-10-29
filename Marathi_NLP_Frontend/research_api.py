"""Research, synthetic-data, dataset, report, and controlled job endpoints."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from reporting import collect_runs, generate_report
from research_core import ExperimentRegistry, LABELS, load_dataset
from experiment_core import adaptive_target_from_training_pool
from synthetic_pipeline import (
    GenerationError, MissingProviderConfiguration, provider_from_environment, run_pipeline,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "results" / "models"
REGISTRY_PATH = ROOT / "results" / "experiment_registry.csv"
LOG_DIR = ROOT / "results" / "logs"
REPORT_PATH = ROOT / "results" / "reports" / "FINAL_RESEARCH_REPORT.md"
MIN_DISK_GIB_MODEL_ONLY = 2.0
MIN_DISK_GIB_RESUMABLE = 4.0
router = APIRouter(prefix="/api")
_job_lock = threading.Lock()
_active_process: subprocess.Popen | None = None
_active_run_id: str | None = None
_external_active_pid: int | None = None


class GenerateRequest(BaseModel):
    language: Literal["Marathi"] = "Marathi"
    task: Literal["hate_speech"] = "hate_speech"
    label: Literal["hate", "offensive", "profane", "not"]
    style: Literal["formal", "informal", "dialectal", "code_mixed", "romanized"] = "informal"
    difficulty: Literal["easy", "medium", "hard", "adversarial"] = "medium"
    count: int = Field(default=4, ge=1, le=12)
    threshold: float = Field(default=0.70, ge=0, le=1)
    duplicate_threshold: float = Field(default=0.88, ge=0.5, le=1)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    model_id: str | None = None
    adaptive_target: bool = False


class ExperimentRequest(BaseModel):
    model: Literal["MahaBERT", "MuRIL", "XLM-RoBERTa", "IndicBERT v2"]
    arm: Literal["A0", "A1", "A2", "A3", "A4", "A5", "A6"] = "A0"
    proportion: Literal[0.05, 0.1, 0.25, 0.5, 1.0] = 0.05
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    epochs: int = Field(default=1, ge=1, le=5)
    batch_size: int = Field(default=2, ge=1, le=4)
    synthetic_file: str | None = None
    synthetic_budget: int = Field(default=0, ge=0, le=500)
    resumable_checkpoints: bool = False


def _read_scope() -> dict[str, Any]:
    return json.loads((ROOT / "configs" / "project_scope.json").read_text(encoding="utf-8"))


def _registry() -> pd.DataFrame:
    return ExperimentRegistry(REGISTRY_PATH).read()


def _reconcile_active_process() -> None:
    global _active_process, _active_run_id, _external_active_pid
    with _job_lock:
        if _active_process is not None and _active_process.poll() is not None:
            _active_process = None
            _active_run_id = None
            _external_active_pid = None
        frame = _registry()
        if frame.empty:
            if _active_process is None:
                _active_run_id = None
                _external_active_pid = None
            return
        running = frame[frame.status.astype(str).str.upper().eq("RUNNING")]
        live_external: tuple[str, int] | None = None
        for row in running.to_dict("records"):
            run_id = str(row.get("experiment_id", ""))
            try:
                # CSV readers may return Windows process IDs as strings such
                # as "7132.0" after mixed/blank rows are inferred as floats.
                pid = int(float(str(row.get("pid")).strip()))
                os.kill(pid, 0)
            except (TypeError, ValueError, OSError):
                ExperimentRegistry(REGISTRY_PATH).update(
                    run_id, "FAILED",
                    error="Training process is no longer running (interrupted). Rerun this configuration or resume a saved optimizer checkpoint.",
                    model=row.get("model"), experiment_arm=row.get("experiment_arm"),
                    output=row.get("output"), resumable_checkpoints=row.get("resumable_checkpoints", False),
                )
                continue
            if _active_process is None:
                live_external = (run_id, pid)
        if _active_process is None:
            if live_external:
                _active_run_id, _external_active_pid = live_external
            elif _external_active_pid is not None:
                _active_run_id = None
                _external_active_pid = None


def _training_is_active() -> bool:
    if _active_process is not None and _active_process.poll() is None:
        return True
    if _external_active_pid is not None:
        try:
            os.kill(_external_active_pid, 0)
            return True
        except OSError:
            return False
    return False


def _load_latest_candidates(limit: int = 250) -> list[dict[str, Any]]:
    files = sorted((ROOT / "results" / "generated_data").glob("*_candidates.jsonl"), reverse=True)
    rows: list[dict[str, Any]] = []
    for path in files:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            continue
        if len(rows) >= limit:
            break
    return rows[-limit:]


def _dataset_manifest() -> dict[str, Any]:
    path = ROOT / "data" / "maha_hate" / "dataset_manifest.json"
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {
            split: {
                "file": Path(info.get("file", "")).name,
                "bytes": info.get("bytes"), "sha256": info.get("sha256"),
                "raw_rows": info.get("raw_rows"), "validated_rows": info.get("validated_rows"),
                "label_counts": info.get("label_counts", {}),
            }
            for split, info in raw.items()
        }
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


@router.get("/scope")
def project_scope() -> dict[str, Any]:
    return _read_scope()


@router.get("/overview")
def overview() -> dict[str, Any]:
    from api import available_models
    models = available_models()
    runs = collect_runs(MODEL_DIR)
    statuses = [str(run.get("status", "UNKNOWN")).upper() for run in runs]
    return {
        "title": "Adaptive Quality-Aware Synthetic Data Generation for Low-Resource Indian Language NLP",
        "research_slice": "Marathi four-class hate-speech detection",
        "dataset": _dataset_manifest(),
        "models": models,
        "configured_models": json.loads((ROOT / "configs" / "models.json").read_text(encoding="utf-8"))["models"],
        "experiments": {"completed": statuses.count("COMPLETED"), "failed": statuses.count("FAILED"),
                        "running": statuses.count("RUNNING"), "not_run": statuses.count("NOT_RUN")},
        "synthetic": _synthetic_stats(),
        "scope": _read_scope(),
    }


def _synthetic_stats() -> dict[str, Any]:
    records = _load_latest_candidates(limit=5000)
    return {
        "records": len(records),
        "accepted": sum(bool(row.get("accepted")) for row in records),
        "rejected": sum(row.get("accepted") is False for row in records),
        "pending_review": sum(bool(row.get("accepted")) and row.get("human_review_required", False) for row in records),
        "status": "live generations only; offline fixtures are excluded",
    }


@router.get("/dataset/stats")
def dataset_stats() -> dict[str, Any]:
    manifest = _dataset_manifest()
    by_label = {label: sum(int(split.get("label_counts", {}).get(label, 0)) for split in manifest.values()) for label in LABELS}
    generated = _load_latest_candidates(limit=5000)
    difficulty = {name: sum(row.get("difficulty") == name for row in generated) for name in ("easy", "medium", "hard", "adversarial")}
    return {"splits": manifest, "total_by_label": by_label,
            "total_rows": sum(int(split.get("validated_rows", 0) or 0) for split in manifest.values()),
            "synthetic": _synthetic_stats(), "difficulty_counts": difficulty}


@router.get("/dataset/download/{split}")
def download_dataset_split(split: Literal["train", "valid", "test"]) -> FileResponse:
    """Download one of the local, allowlisted official dataset split files."""
    path = ROOT / "data" / "maha_hate" / f"hate_{split}.xlsx"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"The local {split} dataset file is missing.")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=path.name,
    )


@router.get("/generated")
def generated_records() -> dict[str, Any]:
    rows = _load_latest_candidates(limit=250)
    return {"records": rows, "count": len(rows), "data_kind": "live API records only; not fixture smoke samples"}


@router.get("/generated/{generation_id}/download")
def download_generated_file(
    generation_id: str,
    kind: Literal["candidates", "accepted"] = "accepted",
) -> FileResponse:
    """Download saved JSONL only for a validated generation ID and known file type."""
    if not re.fullmatch(r"gen_[A-Za-z0-9_-]{1,120}", generation_id):
        raise HTTPException(status_code=422, detail="Invalid generation ID.")
    if kind == "candidates":
        path = ROOT / "results" / "generated_data" / f"{generation_id}_candidates.jsonl"
    else:
        path = ROOT / "data" / "synthetic" / "accepted" / f"{generation_id}_accepted.jsonl"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No saved {kind} file exists for this generation.")
    return FileResponse(path, media_type="application/x-ndjson", filename=path.name)


@router.post("/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    from api import _get_model, available_models
    if training_active():
        raise HTTPException(status_code=409, detail="Synthetic scoring is paused while a training job is using the GPU.")
    try:
        provider, generator_model, validator_model = provider_from_environment()
    except MissingProviderConfiguration as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    models = available_models()
    model_id = request.model_id or (models[0]["id"] if models else None)
    if not model_id:
        raise HTTPException(status_code=503, detail="No completed local classification model is available for the independent classifier check.")
    train_path = ROOT / "data" / "maha_hate" / "hate_train.xlsx"
    if not train_path.is_file():
        raise HTTPException(status_code=409, detail="The official Marathi training split is missing.")
    try:
        train, _ = load_dataset(train_path)
        validation, _ = load_dataset(ROOT / "data" / "maha_hate" / "hate_valid.xlsx")
        validation_texts = set(validation.text)
        train = train.loc[~train.text.isin(validation_texts)].reset_index(drop=True)
        tokenizer, classifier, model_record = _get_model(model_id)

        def predict_candidate(text: str) -> dict[str, Any]:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
            encoded = {key: value.to(classifier.device) for key, value in encoded.items()}
            import torch
            with torch.inference_mode():
                logits = classifier(**encoded).logits[0]
                probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
            order = sorted(probs, reverse=True)
            index = max(range(len(probs)), key=probs.__getitem__)
            return {"label": LABELS[index], "confidence": probs[index], "margin": order[0] - order[1]}

        adaptive_selection = None
        target_label = request.label
        if request.adaptive_target:
            adaptive_selection = adaptive_target_from_training_pool(train, predict_candidate)
            target_label = adaptive_selection["target_label"]

        result = run_pipeline(
            provider=provider, generator_model=generator_model, validator_model=validator_model,
            classifier=predict_candidate, reference_texts=train.text.tolist(),
            label=target_label, count=request.count, style=request.style,
            difficulty=request.difficulty, threshold=request.threshold,
            duplicate_threshold=request.duplicate_threshold, seed=request.seed,
            adaptive_target=request.adaptive_target,
        )
        result["classifier_model"] = model_record["name"]
        result["adaptive_selection"] = adaptive_selection
        return result
    except HTTPException:
        raise
    except (GenerationError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail=f"Synthetic pipeline failed: {exc}") from exc


@router.get("/experiments")
def experiments() -> dict[str, Any]:
    _reconcile_active_process()
    runs = collect_runs(MODEL_DIR)
    registry = _registry()
    experiment_config = json.loads((ROOT / "configs" / "experiments.json").read_text(encoding="utf-8"))
    arm_status = {}
    for arm, description in experiment_config["arms"].items():
        matches = [run for run in runs if str(run.get("experiment_arm", "A0")).upper() == arm]
        arm_status[arm] = {
            "description": description,
            "runs": len(matches),
            "completed": sum(str(r.get("status", "")).upper() == "COMPLETED" for r in matches),
            "failed": sum(str(r.get("status", "")).upper() == "FAILED" for r in matches),
            "status": "COMPLETED" if any(str(r.get("status", "")).upper() == "COMPLETED" for r in matches) else "NOT_RUN",
        }
    return {"runs": runs, "arms": arm_status,
            "registry": registry.fillna("").to_dict("records") if not registry.empty else [],
            "active_run_id": _active_run_id,
            "matrix": {"proportions": experiment_config["proportions"], "seeds": experiment_config["seeds"]}}


def training_active() -> bool:
    _reconcile_active_process()
    return _training_is_active()


def _command_for_run(run_id: str, request: ExperimentRequest, resume_from: str | None = None) -> list[str]:
    script = ROOT / "scripts" / "train_model.py"
    command = [
        sys.executable, str(script), "--model", request.model, "--experiment-arm", request.arm,
        "--experiment-id", run_id, "--proportion", str(request.proportion), "--seed", str(request.seed),
        "--epochs", str(request.epochs), "--batch-size", str(request.batch_size),
    ]
    if request.arm != "A0":
        if not request.synthetic_file:
            raise HTTPException(status_code=422, detail=f"{request.arm} requires a saved synthetic JSONL file.")
        candidate = (ROOT / request.synthetic_file).resolve()
        permitted = [(ROOT / "results" / "generated_data").resolve(), (ROOT / "data" / "synthetic").resolve()]
        if candidate.suffix.lower() != ".jsonl" or not any(candidate.is_relative_to(base) for base in permitted):
            raise HTTPException(status_code=422, detail="Synthetic input must be a JSONL file under results/generated_data or data/synthetic.")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="Synthetic candidate file was not found.")
        if request.synthetic_budget < 1:
            raise HTTPException(status_code=422, detail="A1-A6 require a positive synthetic budget.")
        command += ["--synthetic-jsonl", str(candidate), "--synthetic-budget", str(request.synthetic_budget)]
    if request.resumable_checkpoints:
        command.append("--resumable-checkpoints")
    if resume_from:
        command += ["--resume-from-checkpoint", resume_from, "--resumable-checkpoints"]
    command.append("--evaluate-test")
    return command


@router.post("/experiments/start")
def start_experiment(request: ExperimentRequest) -> dict[str, Any]:
    global _active_process, _active_run_id, _external_active_pid
    _reconcile_active_process()
    with _job_lock:
        if _training_is_active():
            raise HTTPException(status_code=409, detail=f"Experiment {_active_run_id} is already running. Wait for it before starting another.")
        free_bytes = shutil.disk_usage(ROOT).free
        minimum_gib = MIN_DISK_GIB_RESUMABLE if request.resumable_checkpoints else MIN_DISK_GIB_MODEL_ONLY
        if free_bytes < minimum_gib * 1024**3:
            raise HTTPException(
                status_code=507,
                detail=(f"Only {free_bytes / 1024**3:.2f} GiB free. This run needs at least {minimum_gib:.0f} GiB "
                        f"because resumable checkpoints are {'enabled' if request.resumable_checkpoints else 'disabled'}. "
                        "Free disk space or disable optimizer checkpoints before starting."),
            )
        if "api" in sys.modules:
            try:
                sys.modules["api"]._models.clear()
                if hasattr(sys.modules["api"], "torch"):
                    sys.modules["api"].torch.cuda.empty_cache()
            except Exception:
                pass
        run_id = f"{request.model.lower().replace(' ', '-')}_{request.arm.lower()}_{int(time.time())}_{uuid.uuid4().hex[:6]}_seed{request.seed}"
        command = _command_for_run(run_id, request)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{run_id}.log"
        if _active_process is None and "torch" in sys.modules:
            try:
                import gc
                sys.modules["torch"].cuda.empty_cache()
                gc.collect()
            except Exception:
                pass
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           shell=False, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
            _active_process, _active_run_id = process, run_id
            _external_active_pid = None
            return {"status": "RUNNING", "experiment_id": run_id, "pid": process.pid,
                    "log_file": str(log_path.relative_to(ROOT)), "command": command,
                    "warning": "One run at a time. Test evaluation is requested only after validation checkpoint selection."}
        except OSError as exc:
            ExperimentRegistry(REGISTRY_PATH).update(run_id, "FAILED", error=f"Could not start trainer: {type(exc).__name__}: {exc}", model=request.model, experiment_arm=request.arm)
            raise HTTPException(status_code=500, detail=f"Could not start training process: {type(exc).__name__}.") from exc


@router.get("/experiments/{experiment_id}/logs")
def experiment_logs(experiment_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,120}", experiment_id):
        raise HTTPException(status_code=422, detail="Invalid experiment ID.")
    path = LOG_DIR / f"{experiment_id}.log"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No log file was saved for this run.")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"experiment_id": experiment_id, "lines": lines[-120:]}


@router.post("/experiments/{experiment_id}/resume")
def resume_experiment(experiment_id: str) -> dict[str, Any]:
    global _active_process, _active_run_id, _external_active_pid
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,120}", experiment_id):
        raise HTTPException(status_code=422, detail="Invalid experiment ID.")
    _reconcile_active_process()
    with _job_lock:
        if _training_is_active():
            raise HTTPException(status_code=409, detail="Another experiment is still running.")
        registry = _registry()
        rows = registry.loc[registry.experiment_id.astype(str).eq(experiment_id)]
        if rows.empty:
            raise HTTPException(status_code=404, detail="Experiment record not found.")
        row = rows.iloc[0].to_dict()
        if str(row.get("status", "")).upper() not in {"FAILED", "RUNNING"}:
            raise HTTPException(status_code=409, detail="Only failed or interrupted experiments can be resumed.")
        if str(row.get("resumable_checkpoints", "")).lower() not in {"true", "1"}:
            raise HTTPException(status_code=409, detail="This run did not save optimizer state; resume is unavailable. Rerun it from the same saved configuration instead.")
        output = Path(str(row.get("output", ""))).resolve()
        expected_parent = (MODEL_DIR / experiment_id).resolve()
        if output != expected_parent or not output.is_relative_to(MODEL_DIR.resolve()):
            raise HTTPException(status_code=409, detail="Saved run output path failed validation.")
        training_dir = output / "training"
        checkpoints = [p for p in training_dir.glob("checkpoint-*") if (p / "optimizer.pt").is_file()]
        if not checkpoints:
            raise HTTPException(status_code=409, detail="No optimizer checkpoint exists for this run.")
        latest = max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[-1]))
        try:
            request = ExperimentRequest(
                model=str(row.get("model")), arm=str(row.get("experiment_arm", "A0")),
                proportion=float(row.get("proportion", 1.0)), seed=int(row.get("seed", 42)),
                epochs=int(row.get("epochs", 1)), batch_size=int(row.get("batch_size", 2)),
                synthetic_file=row.get("synthetic_file") or None,
                synthetic_budget=int(row.get("synthetic_budget", 0) or 0), resumable_checkpoints=True,
            )
        except Exception as exc:
            raise HTTPException(status_code=409, detail="Saved run configuration is incomplete; rerun from the saved command.") from exc
        command = _command_for_run(experiment_id, request, str(latest))
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{experiment_id}.log"
        with log_path.open("a", encoding="utf-8") as log:
            _active_process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               shell=False, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        _active_run_id = experiment_id
        _external_active_pid = None
        return {"status": "RUNNING", "experiment_id": experiment_id, "pid": _active_process.pid,
                "resumed_from": latest.name, "log_file": str(log_path.relative_to(ROOT))}


@router.post("/report/generate")
def build_report() -> dict[str, str]:
    markdown, csv_path = generate_report(MODEL_DIR, REPORT_PATH)
    return {"markdown": markdown.read_text(encoding="utf-8"), "csv_file": str(csv_path.relative_to(ROOT))}


@router.get("/report")
def read_report() -> dict[str, str]:
    if not REPORT_PATH.is_file():
        generate_report(MODEL_DIR, REPORT_PATH)
    return {"markdown": REPORT_PATH.read_text(encoding="utf-8"), "csv_file": "results/reports/FINAL_RESEARCH_REPORT.csv"}


@router.get("/report/csv")
def download_report_csv() -> FileResponse:
    path = REPORT_PATH.with_suffix(".csv")
    if not path.is_file():
        generate_report(MODEL_DIR, REPORT_PATH)
    return FileResponse(path, media_type="text/csv", filename="marathi_nlp_research_results.csv")
