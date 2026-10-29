# Gemini project handoff

Copy the prompt in the final section into Gemini after opening this existing folder:

`E:\GENAI\CODE\Marathi_NLP_Frontend`

## Project snapshot

- **Project:** Marathi NLP Lab; research implementation for “Adaptive Quality-Aware Synthetic Data Generation for Low-Resource Indian Language NLP.”
- **Frontend/backend:** React + Vite frontend, FastAPI backend, local transformer inference. A legacy Streamlit `app.py` also exists, but React is the primary UI.
- **Current research slice:** Marathi four-class hate-speech classification on L3Cube-MahaHate. Classes map to `hate`, `offensive`, `profane`, and `not` (source labels HATE, OFFN, PRFN, NOT).
- **Current input styles:** Devanagari Marathi, Romanized Marathi, and Marathi-English code-mixed text. These are input styles, not separate trained languages. Gujarati, Assamese, sentiment, NER, and cross-lingual transfer remain unimplemented research scope; do not claim support or results for them.
- **Pages:** Overview, Classify Marathi Text, Research Notes, Model Comparison, Synthetic Lab, Experiments, Dataset Explorer, Research Results, and About & Scope.
- **Start app:** In PowerShell, `Set-Location 'E:\GENAI\CODE\Marathi_NLP_Frontend'; .\scripts\start_project.ps1`. Frontend: `http://127.0.0.1:5173/`; API: `http://127.0.0.1:8000/`; Swagger: `http://127.0.0.1:8000/docs`; OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`.
- **Checks last run from E:** `23 passed` with `.venv\Scripts\python.exe -m pytest -q`; React production build passed with `npm run build` from `frontend`; API health returned `ok`, CUDA, six model entries, and no active training; a recovered MahaBERT checkpoint classified a Marathi sample as `not` with confidence 0.7410.
- **Credentials:** A private `.env` is present and git-ignored. Never paste or upload it, expose its values in logs/UI, or commit it. Do not ask Gemini to reproduce its secrets. Live generation needs a valid provider key and different generator/validator model IDs.
- **Old location:** An inactive duplicate still exists at `C:\Users\Shreyash Chaudhari\Desktop\GENAI\CODE\Marathi_NLP_Frontend`; use the E: folder as the only active project. The old C: duplicate was retained after automatic safety review rejected recursive removal.

## Dataset and results

Local official split files under `data/maha_hate/` are `hate_train.xlsx`, `hate_valid.xlsx`, and `hate_test.xlsx`.

| Split | Raw/validated rows | Per-class counts |
|---|---:|---|
| Train | 21,500 / 21,308 | hate 5,317; offensive 5,296; profane 5,332; not 5,363 |
| Validation | 1,500 / 1,500 | 375 per class |
| Test | 2,000 / 2,000 | 500 per class |

Validated total: 24,808 rows. Split SHA-256 fingerprints and label counts are available at `GET /api/dataset/stats`; hashes and metadata are saved in experiment/run files. There is **currently no dataset file-download API**. The official dataset redistribution license must be checked before exposing source files for download.

Recent saved baselines (test set is report-only and never used to select checkpoints):

| Run/configuration | Validation Macro-F1 | Test Macro-F1 | Test accuracy | Training time |
|---|---:|---:|---:|---:|
| MahaBERT, 5%, 3 epochs, 1,065 real rows, seed 42 | 0.7470 | 0.7423 | 0.7410 | 200.83 s |
| MuRIL, 5%, 3 epochs, 1,065 real rows, seed 42 | 0.5030 | 0.5045 | 0.5585 | 237.35 s |
| MahaBERT A0, 5%, 1 epoch, 1,065 real rows, seed 42, latest clean run `mahabert_a0_1793290925_000da2_seed42` | 0.6456 | 0.6569 | 0.6640 | 83.99 s |
| Recovered MahaBERT A0, 1 epoch, run `mahabert_a0_1793216304_27fe0c_seed42` | 0.6456 | 0.6573 | 0.6645 | Unknown; final serialization ran out of disk, model-only checkpoint was recovered and reevaluated |

Current registry: 12 records, 7 completed and 5 failed. Six saved classifier entries are recognized by the API. There are no completed A1–A6 synthetic-augmentation training comparisons. The single-seed local results are not an independent replication of the paper and do not show that synthetic data improves model quality.

Synthetic pipeline currently records 29 live candidates: 12 accepted by machine checks (all pending human review) and 17 rejected. Accepted synthetic data has not been used for training. Fixture smoke examples are excluded from these counts.

Key artifacts:

- Model checkpoints and detailed metrics: `results/models/<run-id>/best` and `results/models/<run-id>/run.json`.
- Experiment statuses/errors: `results/experiment_registry.csv`; logs: `results/logs/`.
- Candidate records: `results/generated_data/`; accepted pending-review examples: `data/synthetic/accepted/`.
- Generated results: `results/reports/FINAL_RESEARCH_REPORT.md` and `.csv`.
- Scope/method configurations: `configs/project_scope.json`, `configs/models.json`.

## Existing API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | API/device/model/training status |
| GET | `/api/models` | Completed local checkpoints and metrics |
| POST | `/api/predict` | Local classifier inference; JSON `{"text":"...","model_id":"..."}` |
| GET | `/api/scope` | Verified research scope and future work |
| GET | `/api/overview` | Dashboard counts/status |
| GET | `/api/dataset/stats` | Split counts, hashes, synthetic totals; not file download |
| GET | `/api/generated` | Persisted live candidate records |
| POST | `/api/generate` | Live synthetic generation, validation, SDQS, dedupe and persistence |
| GET | `/api/experiments` | Run history, arm counts, active run |
| POST | `/api/experiments/start` | Starts one bounded training run; low-disk guard is enabled |
| GET | `/api/experiments/{experiment_id}/logs` | Saved run log tail |
| POST | `/api/experiments/{experiment_id}/resume` | Resume an eligible optimizer checkpoint |
| POST | `/api/report/generate` | Regenerates report from saved evidence |
| GET | `/api/report` | Markdown report data |
| GET | `/api/report/csv` | Download results CSV |

## Copy/paste prompt for Gemini

```text
You are continuing an existing final-year research project. Work in place in this exact folder: E:\GENAI\CODE\Marathi_NLP_Frontend. Do not create a replacement project, duplicate modules, erase checkpoints, overwrite experiment outputs, or fabricate metrics. Inspect README.md, docs/IMPLEMENTATION_AUDIT.md, docs/FINAL_STATUS_REPORT.md, this handoff, tests, API routes, React pages, run.json files, and the experiment registry before changing anything. The current app is running from E:, so use this folder as the source of truth.

PROJECT FACTS
- Name: Marathi NLP Lab; research theme: adaptive quality-aware synthetic data for low-resource Indian-language NLP.
- Stack: React/Vite frontend, FastAPI backend, PyTorch/Transformers local model inference; legacy Streamlit app also exists but React is the primary application.
- Verified research slice: Marathi four-class L3Cube-MahaHate hate-speech classification. Source labels HATE/OFFN/PRFN/NOT map to hate/offensive/profane/not.
- Input forms currently supported by UI/model workflow: Marathi in Devanagari, Romanized Marathi, and Marathi-English code-mixed text. These are not distinct trained languages. Gujarati, Assamese, sentiment, NER, and cross-lingual transfer are not implemented/evaluated; never claim they are complete.
- Dataset: data/maha_hate/hate_train.xlsx (21,500 raw; 21,308 validated), hate_valid.xlsx (1,500), hate_test.xlsx (2,000); balanced validation/test splits, class counts and SHA-256 are returned by GET /api/dataset/stats. Dataset redistribution permissions must be checked before offering original files for download.
- Actual baselines: MahaBERT 5%/3 epochs/seed 42: validation Macro-F1 0.7470, test Macro-F1 0.7423, accuracy 0.7410, 200.83 sec. MuRIL under matched conditions: validation Macro-F1 0.5030, test Macro-F1 0.5045, accuracy 0.5585, 237.35 sec. A recent clean one-epoch MahaBERT A0 run: test Macro-F1 0.6569, accuracy 0.6640, 83.99 sec. A separate run with the same config recovered from a model-only checkpoint after disk-full final serialization: test Macro-F1 0.6573, accuracy 0.6645; training time unknown. Preserve and distinguish these records.
- Current evidence: 12 registry records (7 completed, 5 failed); six local classifier entries recognized. No A1-A6 synthetic augmentation training comparison is complete. 29 live synthetic records exist (12 machine-accepted, pending human review; 17 rejected); none have been used for training. One seed is not independent verification or statistical significance.
- The private .env contains provider credentials. Never reveal, print, upload, include in a prompt, commit, or overwrite it. Never paste its values into logs or frontend code.
- Start in PowerShell: Set-Location 'E:\GENAI\CODE\Marathi_NLP_Frontend'; .\scripts\start_project.ps1. UI http://127.0.0.1:5173/. API docs http://127.0.0.1:8000/docs. Last checks: Python pytest 23 passed, React npm build passed, API health ok, and actual CUDA prediction succeeded. Use a disk check before training: resumable jobs require at least 4 GiB and model-only jobs at least 2 GiB. Keep the CUDA allocator setting in scripts/start_project.ps1.

CURRENT API ROUTES
GET /api/health, GET /api/models, POST /api/predict, GET /api/scope, GET /api/overview, GET /api/dataset/stats, GET /api/generated, POST /api/generate, GET /api/experiments, POST /api/experiments/start, GET /api/experiments/{experiment_id}/logs, POST /api/experiments/{experiment_id}/resume, POST /api/report/generate, GET /api/report, GET /api/report/csv. OpenAPI docs are at /docs. Dataset statistics exist, but dataset file downloads do not. The result CSV download exists.

WORK TO DO — UPGRADE THE EXISTING PRODUCT
1. Audit the whole app first and produce a prioritized checklist, but do not stop at the checklist: implement all feasible improvements, test each phase, fix failures, update the audit/status report, and summarize remaining blockers.
2. Upgrade the full React UI to a polished, accessible, responsive research workspace. Keep real API data and model inference; remove fake/static result claims. Improve mobile layouts, loading/empty/error states, API connectivity messaging, keyboard and screen-reader support, charts/tables, and clear provenance for metrics.
3. Classifier: replace the single “TRY A SAMPLE” item with a clearly organized sample picker containing several original, human-reviewed illustrative samples for EACH of the four model classes (hate, offensive, profane, not). Cover Devanagari Marathi, Romanized Marathi, and Marathi-English code-mixed styles across the sample library. Include the requested benign example “आजचा दिवस खूप सुंदर आहे…” under NOT. Selecting a sample should only fill the input; the saved local model must produce the prediction and probabilities. Never hardcode the sample's predicted output or imply an illustrative sample is ground truth. Use non-graphic, non-slur examples and clear educational labeling. If “four kinds” is interpreted as four languages, do not silently claim multilingual support: Marathi is the only trained language; first identify datasets, licenses, checkpoints, and evaluation needed for any additional language.
4. Improve dataset and results exports. Add safe, validated, documented API downloads for eligible dataset splits and experiment/results data (CSV and, where useful, XLSX/JSON). For official L3Cube files, verify redistribution terms first; if redistribution is not permitted, provide attribution, metadata/fingerprints, and a source link rather than republishing the files. Add clear UI download controls, correct content types/filenames, split allowlists, path-traversal protection, error handling, and tests. Include report CSV, run registry/results, and accepted synthetic data only with human-review status preserved. Do not export secrets or unintended raw user text.
5. Document every endpoint in OpenAPI with request/response schemas, examples, errors and download behavior. Keep /docs and /openapi.json accurate. Add only needed endpoints, avoid duplicate route implementations, and update the frontend to consume them.
6. Preserve research integrity: validation selects checkpoints; test data is report-only and cannot be used for threshold tuning, sample targeting, checkpoint selection, hyperparameter tuning, or synthetic generation. Compare A0/A4/A6 only with matched model/configuration/splits/budgets where feasible. Do not launch an expensive matrix automatically; inspect GPU/disk, run a small smoke first, and ask before a resource-intensive full matrix.
7. Maintain separate statuses for implemented, unit-tested, integration-tested, model-loaded, fine-tuned, experiment-completed, and independently verified. Never label the entire synopsis complete; only the Marathi task is currently verified. Do not substitute placeholder numbers for missing results.
8. Add tests for all new UI/API flows. Verify end to end: browser → API → files/results → download response. Run Python tests, React production build, smoke tests, and actual API checks; report exact commands and results. Preserve prior experiment files and outputs.

Deliver: working code in this existing E: project, updated docs/status, exact test evidence, current URLs, changed files, actual remaining limitations, and any command needed to resume blocked work. Start by inspecting the repository and current running services, then proceed phase by phase.
```
