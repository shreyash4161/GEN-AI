"""Controlled LLM generation, independent validation, SDQS, deduplication, and persistence."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import requests

from research_core import LABELS, normalize_text, write_record

ROOT = Path(__file__).resolve().parent
PROMPT_VERSION = "marathi-hate-v1"
STYLE_PROMPTS = {
    "formal": "Use formal, standard Marathi in Devanagari.",
    "informal": "Use natural informal Marathi; keep the meaning understandable.",
    "dialectal": "Use a mild, regionally natural Marathi variation without caricature.",
    "code_mixed": "Use natural Marathi-English code-mixing, as in everyday social media.",
    "romanized": "Write Marathi transliterated in Latin letters; do not treat transliteration as a defect.",
}
DIFFICULTY_PROMPTS = {
    "easy": "Make the intended class clear and direct.",
    "medium": "Use realistic context and moderate lexical variation.",
    "hard": "Use subtle context, negation, or indirect wording while preserving the intended label.",
    "adversarial": "Use an edge case such as quotation, negation, or ambiguity. Do not make the label impossible to infer.",
}
SDQS_WEIGHTS = {
    "linguistic_plausibility": 0.15,
    "semantic_consistency": 0.15,
    "label_correctness": 0.25,
    "candidate_diversity": 0.10,
    "train_novelty": 0.10,
    "difficulty_alignment": 0.10,
    "content_relevance": 0.10,
    "hallucination_safety": 0.05,
}


class GenerationError(RuntimeError):
    pass


class MissingProviderConfiguration(GenerationError):
    pass


class OpenAICompatibleProvider:
    """Small Chat Completions client usable with Groq or another compatible endpoint."""

    def __init__(self, api_key: str, base_url: str, timeout: float = 40.0, retries: int = 2):
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def complete_json(self, model: str, system: str, prompt: str) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        response = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    url, headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload, timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                response.raise_for_status()
                body = response.json()
                raw = body["choices"][0]["message"]["content"]
                return parse_json_object(raw)
            except requests.RequestException as exc:
                if attempt < self.retries and (response is None or response.status_code >= 500):
                    time.sleep(min(2 ** attempt, 4))
                    continue
                status = response.status_code if response is not None else "network"
                raise GenerationError(f"LLM provider request failed ({status}). Check endpoint/model/key and retry.") from exc
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise GenerationError("LLM provider returned an invalid JSON response shape.") from exc
        raise GenerationError("LLM provider request exhausted its retry budget.")


def provider_from_environment() -> tuple[OpenAICompatibleProvider, str, str]:
    # Read the project-local .env without adding a runtime dependency. Existing
    # process environment variables intentionally take precedence.
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name and name not in os.environ:
                os.environ[name] = value
    key = os.getenv("GROQ_API_KEY", "").strip()
    generator = os.getenv("GROQ_GENERATOR_MODEL", "").strip()
    validator = os.getenv("GROQ_VALIDATOR_MODEL", "").strip()
    base = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1").strip()
    missing = [name for name, value in (("GROQ_API_KEY", key), ("GROQ_GENERATOR_MODEL", generator), ("GROQ_VALIDATOR_MODEL", validator)) if not value]
    if missing:
        raise MissingProviderConfiguration("Live generation needs " + ", ".join(missing) + ". Set them in the environment or .env; no API key is bundled.")
    if generator.casefold() == validator.casefold():
        raise MissingProviderConfiguration("Generator and validator must be different model IDs for cross-model validation.")
    return OpenAICompatibleProvider(key, base), generator, validator


def parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise GenerationError("LLM response did not contain a JSON object.") from exc
        obj = json.loads(raw[start:end + 1])
    if not isinstance(obj, dict):
        raise GenerationError("LLM response must be a JSON object.")
    return obj


def make_generation_prompt(*, label: str, count: int, style: str, difficulty: str, seed: int, adaptive_target: bool) -> str:
    if label not in LABELS or style not in STYLE_PROMPTS or difficulty not in DIFFICULTY_PROMPTS:
        raise ValueError("Unknown class, style, or difficulty option")
    target = f"Generate {count} distinct Marathi examples for the class `{label}`."
    if adaptive_target:
        target += " These examples target a model uncertainty/error category observed only from permitted training-pool analysis."
    return (
        f"{target}\nLanguage style: {STYLE_PROMPTS[style]}\nDifficulty: {DIFFICULTY_PROMPTS[difficulty]}\n"
        "Do not include slurs or explicit threats unless essential to the requested class; avoid naming real private people. "
        "Do not include explanations in the examples. Do not use validation or test data. "
        f"Use randomization seed {seed} as a prompt hint. Return JSON only: "
        '{"samples":[{"text":"...","label":"..."}]}.'
    )


def generate_candidates(provider: Any, model: str, *, label: str, count: int, style: str,
                        difficulty: str, seed: int, adaptive_target: bool = False) -> list[dict[str, Any]]:
    prompt = make_generation_prompt(label=label, count=count, style=style, difficulty=difficulty,
                                    seed=seed, adaptive_target=adaptive_target)
    result = provider.complete_json(
        model,
        "You create synthetic Marathi social-media text for a research classifier. Follow the requested class but never claim a sample is verified. Output valid JSON.",
        prompt,
    )
    samples = result.get("samples")
    if not isinstance(samples, list):
        raise GenerationError("Generator JSON must include a samples array.")
    output = []
    for item in samples[:count]:
        if not isinstance(item, dict):
            continue
        text = normalize_text(item.get("text", ""))
        label_value = str(item.get("label", label)).strip().lower()
        if text:
            output.append({"text": text, "requested_label": label_value})
    return output


def validate_with_model(provider: Any, model: str, text: str, requested_label: str,
                        requested_difficulty: str) -> dict[str, Any]:
    prompt = (
        "Independently review this synthetic Marathi social-media text. Do not trust its requested label. "
        "Return JSON with keys: label (one of hate/offensive/profane/not), label_confidence (0..1), "
        "linguistic_plausibility (0..1), semantic_consistency (0..1), content_relevance (0..1), "
        "unsupported_claim_risk (0..1), difficulty (easy/medium/hard/adversarial), explanation (short). "
        "Judge language variety fairly: code-mixing and Romanized Marathi are not defects. "
        f"Requested label: {requested_label}; requested difficulty: {requested_difficulty}; text: {json.dumps(text, ensure_ascii=False)}"
    )
    result = provider.complete_json(
        model,
        "You are an independent label and quality validator. The requested label may be wrong. Never infer correctness from the prompt alone.",
        prompt,
    )
    label = str(result.get("label", "")).strip().lower()
    if label not in LABELS:
        raise GenerationError("Validator returned an unknown label.")
    normalized = {"label": label}
    for field in ("label_confidence", "linguistic_plausibility", "semantic_consistency", "content_relevance", "unsupported_claim_risk"):
        try:
            value = float(result[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise GenerationError(f"Validator omitted a numeric {field} score.") from exc
        if not 0 <= value <= 1:
            raise GenerationError(f"Validator {field} must be in [0,1].")
        normalized[field] = value
    normalized["difficulty"] = str(result.get("difficulty", "")).lower()
    if normalized["difficulty"] not in {"easy", "medium", "hard", "adversarial"}:
        raise GenerationError("Validator returned an unknown difficulty category.")
    normalized["explanation"] = normalize_text(result.get("explanation", ""))[:500]
    normalized["validator_type"] = "independent_llm_estimate_not_ground_truth"
    return normalized


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    compact = re.sub(r"\s+", " ", normalize_text(text).casefold())
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[index:index + n] for index in range(len(compact) - n + 1)}


def char_ngram_similarity(left: str, right: str) -> float:
    a, b = _char_ngrams(left), _char_ngrams(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def nearest_similarity(text: str, references: Iterable[str]) -> float:
    return max((char_ngram_similarity(text, ref) for ref in references if normalize_text(ref)), default=0.0)


def compute_sdqs(validator: Mapping[str, Any], requested_label: str, validator_label: str,
                 classifier_label: str, classifier_confidence: float, difficulty_requested: str,
                 difficulty_validated: str, train_similarity: float, candidate_similarity: float) -> tuple[float, dict[str, float]]:
    agrees = validator_label == requested_label == classifier_label
    components = {
        "linguistic_plausibility": float(validator["linguistic_plausibility"]),
        "semantic_consistency": float(validator["semantic_consistency"]),
        "label_correctness": min(float(validator["label_confidence"]), float(classifier_confidence)) if agrees else 0.0,
        "candidate_diversity": 1.0 - candidate_similarity,
        "train_novelty": 1.0 - train_similarity,
        "difficulty_alignment": 1.0 if difficulty_requested == difficulty_validated else 0.45,
        "content_relevance": float(validator["content_relevance"]),
        "hallucination_safety": 1.0 - float(validator["unsupported_claim_risk"]),
    }
    if not all(0 <= score <= 1 for score in components.values()):
        raise ValueError("SDQS components must be normalized to [0,1]")
    if abs(sum(SDQS_WEIGHTS.values()) - 1.0) > 1e-9:
        raise ValueError("SDQS weights must sum to 1")
    return sum(SDQS_WEIGHTS[key] * value for key, value in components.items()), components


def run_pipeline(*, provider: Any, generator_model: str, validator_model: str,
                 classifier: Callable[[str], Mapping[str, Any]], reference_texts: Iterable[str],
                 label: str, count: int, style: str, difficulty: str, threshold: float,
                 seed: int = 42, adaptive_target: bool = False,
                 duplicate_threshold: float = 0.88, output_root: str | Path = ROOT) -> dict[str, Any]:
    if generator_model.casefold() == validator_model.casefold():
        raise ValueError("Generator and validator model IDs must differ.")
    if not 1 <= count <= 12 or not 0 <= threshold <= 1 or not 0 <= duplicate_threshold <= 1:
        raise ValueError("Count or quality thresholds are outside the supported range.")
    generated = generate_candidates(provider, generator_model, label=label, count=count, style=style,
                                    difficulty=difficulty, seed=seed, adaptive_target=adaptive_target)
    if len(generated) < 1:
        raise GenerationError("Generator returned no usable candidate text.")
    references = [normalize_text(text) for text in reference_texts if normalize_text(text)]
    metadata = {
        "provider": "openai_compatible",
        "generator_model": generator_model,
        "validator_model": validator_model,
        "prompt_version": PROMPT_VERSION,
        "seed": seed,
        "language": "Marathi",
        "task": "hate_speech",
        "style": style,
        "controlled_style": True,
        "difficulty_requested": difficulty,
        "adaptive_target": adaptive_target,
    }
    seen: list[str] = []
    records: list[dict[str, Any]] = []
    for item in generated:
        text = item["text"]
        requested = item["requested_label"]
        record: dict[str, Any] = {
            **item,
            "label": requested,
            "generation_metadata": {**metadata, "generated_at": datetime.now(timezone.utc).isoformat()},
            "human_review_required": True,
        }
        rejection = []
        if requested != label:
            rejection.append("generator label did not match requested class")
        try:
            validation = validate_with_model(provider, validator_model, text, label, difficulty)
            prediction = dict(classifier(text))
            classifier_label = str(prediction.get("label", "")).lower()
            classifier_conf = float(prediction.get("confidence", 0))
            if not 0 <= classifier_conf <= 1:
                raise ValueError("classifier confidence must be in [0,1]")
            train_sim = nearest_similarity(text, references)
            candidate_sim = nearest_similarity(text, seen)
            score, components = compute_sdqs(
                validation, label, validation["label"], classifier_label, classifier_conf,
                difficulty, validation["difficulty"], train_sim, candidate_sim,
            )
            if validation["label"] != label:
                rejection.append("independent validator disagreed with requested label")
            if classifier_label != label:
                rejection.append("local classifier disagreed with requested label")
            if classifier_conf < 0.35:
                rejection.append("local classifier confidence below 0.35")
            if train_sim >= duplicate_threshold:
                rejection.append("near duplicate of real training data")
            if candidate_sim >= duplicate_threshold:
                rejection.append("near duplicate of another generated candidate")
            if score < threshold:
                rejection.append("SDQS below configured threshold")
            if validation["unsupported_claim_risk"] >= 0.75:
                rejection.append("validator flagged unsupported or irrelevant content")
            record.update({
                "validated_label": validation["label"],
                "validator": validation,
                "classifier_estimate": {"label": classifier_label, "confidence": classifier_conf},
                "components": components,
                "sdqs": score,
                "duplicate_similarity": max(train_sim, candidate_sim),
                "difficulty": validation["difficulty"],
                "accepted": not rejection,
                "rejection_reasons": rejection,
                "status": "PENDING_HUMAN_REVIEW" if not rejection else "REJECTED",
            })
        except Exception as exc:
            record.update({"accepted": False, "status": "REJECTED", "sdqs": None,
                           "rejection_reasons": [*rejection, f"validation error: {type(exc).__name__}: {exc}"]})
        seen.append(text)
        records.append(record)

    run_id = datetime.now(timezone.utc).strftime("gen_%Y%m%dT%H%M%S_") + uuid.uuid4().hex[:8]
    output_root = Path(output_root).resolve()
    all_path = output_root / "results" / "generated_data" / f"{run_id}_candidates.jsonl"
    accepted_path = output_root / "data" / "synthetic" / "accepted" / f"{run_id}_accepted.jsonl"
    for record in records:
        record["generation_metadata"]["generation_id"] = run_id
        write_record(all_path, record)
        if record["accepted"]:
            write_record(accepted_path, record)
    return {
        "generation_id": run_id,
        "requested": count,
        "generated": len(records),
        "accepted": sum(bool(r["accepted"]) for r in records),
        "rejected": sum(not bool(r["accepted"]) for r in records),
        "acceptance_rate": sum(bool(r["accepted"]) for r in records) / len(records),
        "threshold": threshold,
        "sdqs_weights": SDQS_WEIGHTS,
        "candidate_file": str(all_path.relative_to(ROOT)) if all_path.is_relative_to(ROOT) else str(all_path),
        "accepted_file": (str(accepted_path.relative_to(ROOT)) if accepted_path.is_relative_to(ROOT) else str(accepted_path)) if any(r["accepted"] for r in records) else None,
        "records": records,
        "data_status": "LIVE_LLM_OUTPUT_PENDING_HUMAN_REVIEW",
    }
