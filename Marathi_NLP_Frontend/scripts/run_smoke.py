"""Offline smoke test for the data, quality, and result-recording path."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_core import ExperimentRegistry, SDQSConfig, assert_no_split_leakage, load_dataset, score_candidate, stratified_subset

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    source = root / "fixture.csv"
    pd.DataFrame({"text": [f"मराठी नमुना {i}" for i in range(20)], "label": ["HATE"] * 10 + ["NOT"] * 10}).to_csv(source, index=False)
    frame, metadata = load_dataset(source)
    subset = stratified_subset(frame, .5, seed=42)
    selected = set(subset.text)
    complement = frame.loc[~frame.text.isin(selected)]
    assert_no_split_leakage(subset, complement)
    cfg = SDQSConfig({"linguistic": .25, "semantic": .20, "label": .30, "novelty": .15, "relevance": .10}, threshold=.5)
    candidate = score_candidate("ही एक मराठी चाचणी आहे", "not", frame.text, cfg,
                                validator=lambda _: ("not", .9), semantic_score=.9,
                                metadata={"source": "fixture", "provider": "none"})
    assert candidate["accepted"]
    registry = ExperimentRegistry(root / "results" / "experiment_registry.csv")
    registry.update("offline-smoke", "COMPLETED", model="none", synthetic_count=1)
    assert registry.read().iloc[0].status == "COMPLETED"
    (root / "candidate.json").write_text(json.dumps(candidate, ensure_ascii=False), encoding="utf-8")
    assert json.loads((root / "candidate.json").read_text(encoding="utf-8"))["accepted"]
    print(json.dumps({"status": "PASS", "dataset_rows": metadata["rows"], "subset_rows": len(subset), "candidate_accepted": candidate["accepted"], "report_roundtrip": "PASS", "provider": "fixture only; not newly generated research data"}, ensure_ascii=False))
