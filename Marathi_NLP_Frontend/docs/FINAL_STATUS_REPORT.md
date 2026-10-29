# Final project status

Status date: 2026-10-29. This report separates built code, executed checks, trained models, and research results.

## Relocation verification (2026-10-29)

The active project copy is `E:\GENAI\CODE\Marathi_NLP_Frontend`. The project files and saved results were copied from C:; all 21,680 source files checked were present at the E: destination with matching sizes except the E:-specific startup script, which includes the CUDA allocator setting. The recovered checkpoints' Windows junctions were recreated on E:. From the E: copy, 23 tests passed, the React production build passed, both local services returned healthy responses, and the recovered MahaBERT model returned a CUDA prediction (`not`, confidence 0.7410). The previous C: copy remains as an inactive duplicate because automatic safety review rejected its recursive removal.

## Scope and implemented application

The primary app is a React + Vite frontend backed by FastAPI. Its pages are Overview, four-class Marathi classifier, Synthetic Lab, Model Comparison, Experiments, Dataset Explorer, Research Report, and About & Scope. The API loads only completed checkpoints with full validation results and runs transformer inference on the local machine. The separate legacy Streamlit app remains available.

The verified research slice is only Marathi four-class hate-speech detection on L3Cube-MahaHate. The code also implements a controlled synthetic generation interface, distinct generator/validator model checks, local classifier agreement, eight weighted SDQS components, novelty/deduplication checks, candidate/accepted JSONL persistence, A0-A6 data selection, bounded experiment launching, run logs/resume endpoints, and factual Markdown/CSV reporting. Live LLM candidates are tracked in the generated report; accepted examples remain pending human review and have not been used for training. Gujarati, Assamese, sentiment analysis, NER, and cross-lingual transfer are scope interfaces only; no results are claimed.

## Verification ledger

| State | Evidence / status |
|---|---|
| Python environment | Python 3.12.10; PyTorch 2.14.1+cu132; CUDA available; NVIDIA RTX 4050 Laptop GPU with 6 GiB |
| Unit/API tests | 23 passed (`.venv\\Scripts\\python.exe -m pytest -q`), including research routes, provider isolation, and low-disk run rejection |
| Synthetic end-to-end smoke | Passed with fixture-only provider: candidate generation → independent validator → local classifier estimate → SDQS → duplicate rejection → JSONL persistence. Not live research output |
| React production build | Passed with Vite 6.4.5 (`npm run build` in `frontend`) |
| Running frontend | HTTP 200 at `http://127.0.0.1:5173/` |
| Running API | HTTP 200 at `http://127.0.0.1:8000/api/health`; 5 model entries; CUDA; no active training |
| Actual model load/inference | Recovered MahaBERT checkpoint loaded through `/api/predict`; Marathi sample returned `not` with confidence 0.7410 and four-class probabilities |
| Model fine-tuning | MahaBERT and MuRIL completed previously and checkpoints are present |
| A0 baselines | MahaBERT and MuRIL were trained on matched 5% subsets (1,065 rows), seed 42, 3 epochs, batch size 2, same full validation/test splits |
| A1-A6 training | Not run locally. Those arms are available through code/UI but need persisted accepted candidates and actual training runs |
| Live LLM provider | Configured from the private project `.env`; five live Groq generation/validation calls are persisted. The API key is not included in this report |
| Live synthetic candidate | Regenerated report currently records 29 candidates: 12 machine-accepted and pending human review, 17 rejected. None have been used for training. |
| Independent result verification | Not completed; one seed per baseline and no external replication/significance test |
| Browser interaction for newly added pages | Not automated in this turn; production build and page API endpoints were verified. Prior classifier browser-to-API exercise predates these page additions |

## Measured local baseline runs

| Model | Validation Macro-F1 | Test Macro-F1 | Test accuracy | Train rows | Training time |
|---|---:|---:|---:|---:|---:|
| MahaBERT | 0.7470 | 0.7423 | 0.7410 | 1,065 | 200.83 s |
| MuRIL | 0.5030 | 0.5045 | 0.5585 | 1,065 | 237.35 s |

The screenshot's MahaBERT A0 one-epoch run was recovered from its saved model-only checkpoint after final serialization failed for lack of disk space. It has validation Macro-F1 0.6456, test Macro-F1 0.6573, accuracy 0.6645, and unknown training duration. Recovery is recorded in `run.json`; it is not reported as a clean training save. Its `best` path is a Windows junction to `training/checkpoint-533`.

MahaBERT minus MuRIL on these matched single-seed runs is +0.2378 test Macro-F1, +0.1825 accuracy, and 36.51 seconds lower recorded training time. These values are the saved trainer outputs, not an independent replication. The paper-reported scores use a different setup and the local results are not a reproduction.

## Recovered UI-launched run

MahaBERT A0, one epoch, 1,065 real rows: validation Macro-F1 0.6456, test Macro-F1 0.6573, and accuracy 0.6645. Training reached step 533 and completed full validation, then optimizer-state serialization failed. The saved model-only checkpoint was re-evaluated on validation and then test without retraining. Training time is unavailable. The `best` path is a Windows junction to the saved checkpoint because disk space was low.

## Run experiment preview and launch

From PowerShell at `C:\\Users\\Shreyash Chaudhari\\Desktop\\GENAI\\CODE\\Marathi_NLP_Frontend`:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_experiments.py --models MahaBERT --arms A0 --proportions 0.05
```

That command previews only. Launch a single bounded run by adding `--execute`. A1-A6 additionally require a saved synthetic JSONL path and positive `--synthetic-budget`; matrices larger than six runs require an explicit acknowledgement. Training from the UI allows at most 5 epochs and batch size 4, checks disk space, and only runs one process at a time.

## Run the application

If the servers have stopped, run `.\scripts\start_project.ps1` from the project root. It starts both services in the background and verifies them.

```powershell
.\\.venv\\Scripts\\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

```powershell
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`. The API docs are at `http://127.0.0.1:8000/docs`.

For live synthetic generation, copy `.env.example` to `.env` or set `GROQ_API_KEY`, `GROQ_GENERATOR_MODEL`, and `GROQ_VALIDATOR_MODEL` in the process environment; generator and validator IDs must be distinct. Do not commit credentials.

## Outputs and limitations

- Model checkpoints and complete run metrics: `results/models/<run-id>/best` and `results/models/<run-id>/run.json`.
- Experiment statuses and errors: `results/experiment_registry.csv`; logs: `results/logs/`.
- Synthetic candidate logs: `results/generated_data/`; accepted pending-review data: `data/synthetic/accepted/`.
- Live candidate totals and experiment evidence are regenerated from persisted records in `results/reports/FINAL_RESEARCH_REPORT.md` and `.csv`.
- Report: `results/reports/FINAL_RESEARCH_REPORT.md` and `.csv`.
- No local A0/A4/A6 augmentation comparison is complete. No synthetic improvement is claimed.
- The training screen does not imply research equivalence: XLM-R/IndicBERT have not been loaded or trained here, and IndicBERT's configured encoder requires a task classification head.
- Accepted synthetic records are machine estimates requiring human review. The held-out test split is not used by the generation pipeline or SDQS thresholds.
- The synopsis-wide multilingual and multi-task project remains incomplete; the working implementation is the Marathi hate-speech slice plus reusable workflow interfaces.






