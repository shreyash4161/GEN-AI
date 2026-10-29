import json

import pandas as pd

from experiment_core import adaptive_target_from_training_pool
import synthetic_pipeline
from synthetic_pipeline import run_pipeline


class FixtureProvider:
    """Deterministic test provider; never represents live research generation."""
    def complete_json(self, model, system, prompt):
        if model == "generator-test":
            return {"samples": [
                {"text": "आजचा दिवस खूप सुंदर आहे आणि मन प्रसन्न झाले.", "label": "not"},
                {"text": "आजचा दिवस खूप सुंदर आहे आणि मन प्रसन्न झाले.", "label": "not"},
            ]}
        return {
            "label": "not", "label_confidence": .95, "linguistic_plausibility": .92,
            "semantic_consistency": .96, "content_relevance": .9,
            "unsupported_claim_risk": .02, "difficulty": "medium",
            "explanation": "Fixture-only validator result",
        }


def test_synthetic_pipeline_validates_deduplicates_and_persists(tmp_path):
    result = run_pipeline(
        provider=FixtureProvider(), generator_model="generator-test", validator_model="validator-test",
        classifier=lambda _: {"label": "not", "confidence": .94}, reference_texts=["असंबंधित प्रशिक्षण मजकूर"],
        label="not", count=2, style="informal", difficulty="medium", threshold=.7,
        duplicate_threshold=.88, output_root=tmp_path,
    )
    assert result["generated"] == 2
    assert result["accepted"] == 1
    assert result["rejected"] == 1
    assert result["data_status"] == "LIVE_LLM_OUTPUT_PENDING_HUMAN_REVIEW"
    candidate_path = tmp_path / result["candidate_file"]
    accepted_path = tmp_path / result["accepted_file"]
    candidates = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines()]
    accepted = [json.loads(line) for line in accepted_path.read_text(encoding="utf-8").splitlines()]
    assert len(candidates) == 2
    assert sum(row["accepted"] for row in candidates) == 1
    assert accepted[0]["human_review_required"] is True
    assert candidates[1]["duplicate_similarity"] >= .88


def test_adaptive_target_uses_lowest_recall_on_supplied_train_pool_only():
    frame = pd.DataFrame({
        "text": [f"{label} example {index}" for label in ("hate", "offensive", "profane", "not") for index in range(4)],
        "label": [label for label in ("hate", "offensive", "profane", "not") for _ in range(4)],
    })
    result = adaptive_target_from_training_pool(
        frame, lambda text: {"label": "offensive" if text.startswith("hate ") else text.split()[0]},
        per_class=4, seed=17,
    )
    assert result["target_label"] == "hate"
    assert result["per_class_recall"]["hate"] == 0
    assert result["per_class_recall"]["not"] == 1
    assert result["split_used"] == "train_only"


def test_provider_configuration_reads_project_dotenv_without_printing_values(tmp_path, monkeypatch):
    monkeypatch.setattr(synthetic_pipeline, "ROOT", tmp_path)
    for name in ("GROQ_API_KEY", "GROQ_API_BASE", "GROQ_GENERATOR_MODEL", "GROQ_VALIDATOR_MODEL"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=private-test-value\nGROQ_API_BASE=https://example.test/v1\n"
        "GROQ_GENERATOR_MODEL=generator-id\nGROQ_VALIDATOR_MODEL=validator-id\n",
        encoding="utf-8",
    )
    provider, generator, validator = synthetic_pipeline.provider_from_environment()
    assert provider._api_key == "private-test-value"
    assert provider.base_url == "https://example.test/v1"
    assert (generator, validator) == ("generator-id", "validator-id")


def test_process_environment_overrides_project_dotenv(tmp_path, monkeypatch):
    monkeypatch.setattr(synthetic_pipeline, "ROOT", tmp_path)
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=local-key\nGROQ_GENERATOR_MODEL=local-generator\nGROQ_VALIDATOR_MODEL=local-validator\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GROQ_API_KEY", "process-key")
    monkeypatch.setenv("GROQ_GENERATOR_MODEL", "process-generator")
    monkeypatch.setenv("GROQ_VALIDATOR_MODEL", "process-validator")
    provider, generator, validator = synthetic_pipeline.provider_from_environment()
    assert provider._api_key == "process-key"
    assert (generator, validator) == ("process-generator", "process-validator")
