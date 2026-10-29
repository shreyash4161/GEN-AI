# Implementation audit

Audit date: 2026-10-28. Scope: the checked out repository, the saved notebook source and its embedded outputs. The notebook was not executed during this audit.

## Repository contents and evidence

The notebook `Adaptive_Marathi_NLP_Final.ipynb` is one directory above this project. The project contains official MahaHate XLSX splits, trained CUDA checkpoints for MahaBERT and MuRIL, JSON run metadata, an experiment registry/report, a FastAPI inference service, and a React frontend. The notebook's `/content` outputs remain separate from these local artifacts.

| Area | Found | Verified state / gaps |
|---|---|---|
| MahaHate loading | `scripts/prepare_data.py` fetches official train/valid/test XLSX; core loader maps labels, drops conflicting text groups, records fingerprints. | Executed locally: 21,500 raw train rows, 21,308 cleaned train rows, validation 1,500, test 2,000. Trainer removes 11 exact train/validation overlaps before subset selection. |
| Subsets | Stratified 5%, 10%, 25% with seed 42 in quick mode. | Present in notebook; 50% and 100% are excluded by `PRIMARY_ONLY=True`. |
| MahaBERT | `l3cube-pune/marathi-bert` base encoder, four-class head, best checkpoint selected by validation Macro-F1. | Loaded and fine-tuned locally on CUDA using 5% (1,065 rows), three epochs. Validation Macro-F1 0.7470; test Macro-F1 0.7423, accuracy 0.7410. Saved checkpoint loaded by API and a real React browser inference passed. |
| MuRIL | `google/muril-base-cased`, same classifier/trainer setup. | Loaded/fine-tuned with matched 5% subset, seed, epochs, batch size, and full splits. Validation Macro-F1 0.5030; test Macro-F1 0.5045, accuracy 0.5585. |
| XLM-R / IndicBERT | Configured in `configs/models.json`; trainer accepts both names. | Not loaded or trained. IndicBERT MLM-only requires a newly trained classification head. |
| Generation | Groq client, structured JSON prompts, retries, generated CSV output, optional key prompt. | Embedded output records candidate generation and API 400/JSON-validation failures. Credentials must be supplied at run time. This local environment did not run generation. |
| SDQS | Notebook uses linguistic heuristic, embedding similarity, optional LLM review, classifier diagnostic and weighted score; SDQS threshold selection uses validation. | Useful prototype. The configured “diversity” ingredient is not an intrinsic per-example measurement and candidate acceptance depends on optional model outputs. The new core makes missing independent label validation a rejection and records the limitation. |
| Deduplication | Exact and embedding-similarity checks against train/candidate rows. | In notebook. No cross-split synthetic audit was reproducible locally. |
| A0–A6 | Definitions and experiment registry logic exist in notebook; generation pools are separate per percentage/seed. | Embedded validation output lists 21 runs: A0–A5 have some runs; A6 has no accepted candidates at each of 5%, 10%, 25%. Actual test cell has no saved output. Results in those notebook cells are validation outputs; cannot answer held-out test comparisons. |
| Difficulty/adaptive | Confidence/error/subgroup-driven selection and difficulty prompt tags. | Partial. Difficulty tags are generation instructions, not validated difficulty labels. Adaptive rows have zero accepted samples in saved registry output, so no adaptive retraining experiment completed. |
| React and API | `frontend/` Vite/React, `api.py` FastAPI. API scans completed runs with full validation; model choice sorts by validation Macro-F1; local inference returns all four softmax probabilities. | Vite production build passed. Chrome browser submitted a Marathi sentence through React to FastAPI; MahaBERT returned “Not abusive”, all four probabilities summed to 1, and no browser errors occurred. API health 200; whitespace input correctly returned 422. |
| Retained Streamlit interface | Existing `app.py` still supports legacy classify/comparison/results. | Kept for compatibility, but React is the primary app. The old Streamlit server was stopped after migration. |
| Tests and docs | Core tests, React build, run-based report, and setup/audit docs are present. | 14 unit tests pass; fixture smoke, Python compile, production build, real HTTP inference, and real browser inference verified. |

## Notebook result limitations

The embedded notebook experiment registry shows A0/A1/A2/A3/A4/A5 validation runs at selected percentages, but A6 was `NOT RUN: no accepted candidates`. The notebook's final test evaluation cell has no saved output. Separate local 5% MahaBERT and MuRIL runs now have held-out test evaluations; they use the same local trainer conditions but do not fill the missing A0/A4/A6 test comparison. Do not combine notebook validation values with these local runs.

The visible candidate-generation output includes successfully returned rows, alongside Groq 400 errors. Generated examples are API output and not necessarily persisted locally. The saved notebook has no locally available environment lock or exact hardware/software metadata for its earlier Colab run.

## New shared utilities

`research_core.py` implements CSV/TSV/JSON/JSONL/XLSX loading, the canonical four-label mapping, safe invalid-row handling, text normalization, duplicate statistics, fingerprints, stratified subsets, exact split-leakage checks, confidence-based difficulty assignment, explicit SDQS weight validation, candidate validation records, append-only JSONL persistence, and a CSV experiment registry. SDQS components are bounded and documented in code. The fallback linguistic and relevance components are heuristics; semantic similarity is zero unless supplied; and a candidate cannot pass without an independent validator callback that agrees with the requested label.

The model config lists MahaBERT, MuRIL, XLM-R, and IndicBERT. `scripts/train_model.py` supports one model at a time, validation Macro-F1 best-checkpoint selection, saved run metadata, and opt-in test evaluation. MahaBERT and MuRIL completed local runs; XLM-R and IndicBERT remain unverified. The supplied paper reports MahaBERT and MahaRoBERTa 4-class Macro-F1 values of 0.802 and 0.803 under its own setup; local MahaBERT is 0.7423 on one 5% run and is not a paper replication. The local trainer normalizes Unicode/whitespace and deduplicates rows, but does not exactly reproduce the paper's baseline tweet-cleaning pipeline.

## Local execution constraints

At audit time: Python 3.12.10; project venv has PyTorch 2.14.1+cu132 with CUDA on the RTX 4050 Laptop GPU (6 GiB). Official data and two full-validation checkpoints are present. No Groq key is configured, so LLM candidate generation was not run. Checks: 14 tests pass; fixture smoke, Python compile, React production build, API endpoint, and real browser-to-model inference pass. The report has eight saved records (three completed, five failed); the 64-row run is a wiring smoke, not a benchmark. MahaBERT and MuRIL each have one seed; independent reruns and significance tests are absent.

## Implementation update (2026-10-28)

### Newly wired functionality

- Added `configs/project_scope.json` to mark the only verified language/task pair (Marathi hate speech) and clearly identify future-work pairs.
- Added `experiment_core.py` for deterministic A0-A6 synthetic selection, training-pool deduplication, difficulty filtering, and adaptive-target filtering.
- Added `synthetic_pipeline.py` for controlled Marathi style/difficulty prompts, distinct-model validation, local classifier agreement, weighted SDQS, near-duplicate checks, and append-only candidate/accepted JSONL output. Accepted items remain `PENDING_HUMAN_REVIEW`.
- Extended `scripts/train_model.py` with experiment-arm synthetic input, saved configuration, optional optimizer checkpoints/resume, and richer held-out metrics. `scripts/run_experiments.py` previews by default and requires explicit bounded execution.
- Added research API routes for scope, overview, dataset statistics, generation, run start/log/resume, experiments, and report build/download.
- Expanded React navigation to eight pages, including responsive synthetic, experiment, dataset, comparison, and report workflows.
- Added tests for research endpoints, absent provider configuration, and a complete fixture pipeline through acceptance/rejection and persistence.
- A6 now selects the lowest-recall label from a deterministic 16-per-class sample of the cleaned training pool and returns that evidence; the validation and test files are not inputs to target selection.
- Private `.env` loading now uses process-environment precedence; `.env` is ignored by Git. One live Groq request (GPT-OSS 120B generator, GPT-OSS 20B validator) completed end to end: 1 candidate, SDQS 0.9779, accepted, persisted to candidate and accepted JSONL, marked pending human review. The key was not included in tool output or status records.

### Current evidence after this update

- 22 tests pass; fixture smoke passes; Python environment reports CUDA/RTX 4050; React production build passes.
- New API process is running locally, reports four completed checkpoints, and successfully loaded MahaBERT for a real `/api/predict` request. Vite returns HTTP 200.
- Fifteen live provider candidates now exist across five generation IDs: 9 accepted pending human review and 6 rejected. One four-candidate batch came from an unisolated test; the test is now isolated. No A1-A6 fine-tuning was performed. The new React pages were build/API checked but not browser-automated after this update.
- Follow-up experiment comparisons must use the same model configuration, seed, real-data subset, validation/test splits, and comparable synthetic budget. Current saved local baselines have no augmentation arms.



- Verification update: an API test initially read the configured .env and unintentionally submitted a second, four-candidate live request. Those four outputs are retained in the generated-data/accepted folders, all marked pending human review and not used for training. The test now points dotenv loading at a temporary directory and cannot call the provider. The report now counts persisted live records.



- UI-started MahaBERT A0 run `mahabert_a0_1793215515_fceb14_seed42` reached step 533 and recorded full validation (Macro-F1 0.6456), but optimizer-state serialization failed. Its saved model-only checkpoint was re-evaluated on validation and then test without retraining: test Macro-F1 0.6573, accuracy 0.6645. Run metadata preserves the original error and records missing training time. `best` uses a Windows junction to the checkpoint because C: had only ~0.25 GiB free after the serialization attempt. A PID parser fix now handles values such as 7132.0; a regression test covers recovery after API restart.


- The UI-started A0 run `mahabert_a0_1793216304_27fe0c_seed42` finished training and full validation, but final model serialization failed with `SafetensorError: ... There is not enough space on the disk (os error 112)`. The intact model-only checkpoint at `training/checkpoint-533` was re-evaluated on validation and held-out test. Test Macro-F1 is 0.6573 and test accuracy is 0.6645. Training duration is unavailable; this was checkpoint recovery, not a fresh successful final save.
- Its `best` entry is a Windows junction to the recovered checkpoint. C: had approximately 1.27 GiB free during this fix.
- Added `scripts/start_project.ps1`; it starts the API and Vite hidden if either service is down, waits for health checks, and prints URLs. Verified API health and frontend HTTP 200.
- Training now checks disk space before spawning: resumable runs require at least 4 GiB free and model-only runs require 2 GiB. Research pages now show the API URL and startup command instead of generic `Failed to fetch` when the backend is unreachable.
- Verification on 2026-10-29: Python compilation passed; 23 tests passed; React production build passed; API health and frontend HTTP 200 passed. The recovered checkpoint loaded through `/api/predict` and returned four-class scores for a Marathi sample. Five model entries are available and no training job is active. The refreshed report records 29 candidates (12 machine-accepted, 17 rejected); accepted candidates remain pending human review.
- Relocation: the active project is now `E:\GENAI\CODE\Marathi_NLP_Frontend`. The E: startup script enables PyTorch expandable CUDA memory segments; after this setting, the recovered model loaded on CUDA and `/api/predict` succeeded. The prior C: project remains inactive because the safety review blocked its recursive removal.
