# Marathi NLP Lab

Research code for the implemented Marathi hate-speech slice of **Adaptive Quality-Aware Synthetic Data Generation for Low-Resource Indian Language NLP**. The current research dataset is L3Cube-MahaHate with four classes: HATE, OFFN, PRFN, and NOT. Gujarati, Assamese, sentiment analysis, NER, and cross-lingual transfer remain future work.

## What is in this repository

- `Adaptive_Marathi_NLP_Final.ipynb` (one level above this folder): prior Colab notebook containing data loading, MuRIL fine-tuning, Groq candidate generation, SDQS signals, validation-based filtering, and A0–A6 orchestration.
- `research_core.py`, `experiment_core.py`, `synthetic_pipeline.py`: dataset/leakage helpers, deterministic A0–A6 candidate selection, controlled generation prompts, independent LLM validation, multi-signal SDQS, duplicate checks, and persistent JSONL records.
- `api.py`, `research_api.py`: local FastAPI inference, dataset, experiment, report, and synthetic-lab endpoints. Training jobs are bounded to one at a time and can save resumable optimizer checkpoints.
- `frontend/`: responsive React research workspace with overview, four-class classifier, synthetic lab, model comparison, bounded experiment launcher, dataset explorer, report download, and scope/about pages.
- `app.py`: retained Streamlit interface for compatibility; the primary user interface is now React.
- `configs/`: canonical model identifiers, label order, experiment arms, and proportions.
- `scripts/`: environment inspection and an offline fixture smoke test.
- `docs/IMPLEMENTATION_AUDIT.md` and `docs/FINAL_STATUS_REPORT.md`: evidence-based audit and status.

## State of the evidence

The React interface calls a local FastAPI endpoint, which performs inference with saved transformer checkpoints; it does not return hard-coded predictions. MahaBERT (`l3cube-pune/marathi-bert`) is the Marathi-specific base model named in the supplied paper. MuRIL is also retained as a comparison. The app ranks defaults by validation Macro-F1, never by held-out test metrics. Matched 5% runs use 1,065 examples, seed 42, three epochs, the same full validation/test splits, and batch size 2. MahaBERT obtained test Macro-F1 0.7423 / accuracy 0.7410 in 200.83 seconds; MuRIL obtained 0.5045 / 0.5585 in 237.35 seconds. See each run's `run.json` for complete metrics. This one-seed setup is a low-resource baseline, not a replication of the paper or evidence that synthetic data improved results.

The supplied 2022 paper reports four labels and balanced official splits (21,500 train / 1,500 validation / 2,000 test). Its reported four-class Macro-F1 values include 0.802 for MahaBERT and 0.803 for MahaRoBERTa. Those figures use the paper's experiment setup and must not be treated as directly comparable with local runs. See Research notes in the UI. Reference: Velankar et al., [arXiv:2203.13778](https://arxiv.org/abs/2203.13778).

Live synthetic generation requires provider credentials and distinct generator/validator model identifiers. Configure them from `.env.example` (never commit `.env`); accepted records are estimates and remain marked for human review. The offline smoke test uses fixtures only and is excluded from research counts. The shared SDQS utility requires an independent validator and agreement from the local classifier to accept; its quality signals are model estimates/heuristics, not expert language judgment.

## Windows setup

PowerShell from this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\prepare_data.py
python scripts\validate_environment.py
python -m pytest -q
python scripts\run_smoke.py
```

Run the local inference API and React app in separate PowerShell terminals. In both terminals, first `cd` to this project folder and activate `.venv`.

Terminal 1 (model API):

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Terminal 2 (React interface):

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The interface uses local checkpoints and the API reports whether the backend can find a completed model. The API loads a selected local checkpoint the first time it is used. To keep using the older Streamlit interface, run `streamlit run app.py` instead.

The React app's Synthetic Lab reads the project-root `.env` file for `GROQ_API_KEY`, `GROQ_GENERATOR_MODEL`, and `GROQ_VALIDATOR_MODEL` (model IDs must differ). `.env` is git-ignored. Add your key locally; never commit or share it. A6 selects a target class from lowest recall on a deterministic sample of the training split only; it reports the sample's recall evidence. On accepted output, copy the saved `data/synthetic/accepted/...jsonl` path into the Experiments form and select A1–A6. The experiment launcher restricts training to 1–5 epochs and batch sizes 1–4 and checks free disk before launch. For a preview without starting training:

```powershell
python scripts\run_experiments.py --models MahaBERT --arms A0 --proportions 0.05
```

The report page rebuilds `results/reports/FINAL_RESEARCH_REPORT.md` and downloads the associated CSV from saved run metadata only.

For GPU training, install a CUDA-enabled PyTorch wheel after the other requirements. The verified local build on an RTX 4050 was:

```powershell
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu132
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then activate again. The `py` launcher can select another installed Python version; Python 3.10–3.12 is recommended for the pinned Transformers stack.

## Data and model runs

Place the official files under `data/maha_hate/`:

```text
data/maha_hate/hate_train.xlsx
data/maha_hate/hate_valid.xlsx
data/maha_hate/hate_test.xlsx
```

`research_core.load_dataset` accepts CSV, TSV, JSON, JSONL, XLSX, and XLS. It maps the official abbreviations and rejects unrecognized labels from the returned labeled frame while reporting invalid-row counts. Keep raw files untouched. Check exact duplicate overlap across train and validation before training. The notebook fetches train/validation before it loads test in its final evaluation section; preserve that sequencing if running it.

`scripts\prepare_data.py` downloads the official splits and validates their schema. It does not overwrite existing split files. The core loader removes within-file duplicate text, drops all copies of conflicting-label text groups, and the trainer removes exact train/validation overlaps before sampling. The test split is only read by the trainer when `--evaluate-test` is supplied after validation-based checkpoint selection.

The intended checkpoints are in `configs/models.json`. MahaBERT and MuRIL are trained through the same four-class trainer; XLM-R and IndicBERT v2 remain configured but are not trained here. The unified trainer supports one model run at a time; the full experiment matrix does not start automatically. Example 5% runs:

```powershell
python scripts\train_model.py --model MuRIL --proportion 0.05 --epochs 3 --batch-size 2 --evaluate-test
python scripts\train_model.py --model MahaBERT --proportion 0.05 --epochs 3 --batch-size 2 --evaluate-test
```

For Groq candidate generation, copy `.env.example` to `.env` or set the environment variable in PowerShell (do not commit `.env`):

```powershell
$env:GROQ_API_KEY = "your-key"
```

The notebook may prompt interactively when the variable is absent. The shared core itself does not call an LLM. `scripts\run_smoke.py` uses local fixtures only, makes no paid API calls, and does not create research results.

## Results and app

The trainer saves model/tokenizer files and `run.json` under `results/models/<run-id>/best`. FastAPI discovers only completed checkpoints with a full validation split and returns live model probabilities; React displays those results and the saved metrics. Failures and interrupted runs remain in `results/experiment_registry.csv`. The report generator writes `results/reports/FINAL_RESEARCH_REPORT.md` and `.csv` from saved run and registry records; missing scores remain blank.

Start the legacy Streamlit compatibility UI (React is the primary frontend) with:

```powershell
streamlit run app.py
```

Generate a factual report from saved run metadata:

```powershell
python scripts\generate_report.py
```

This writes `results\reports\FINAL_RESEARCH_REPORT.md` and `.csv`. If no model runs are saved, it reports unavailable metrics and leaves score fields blank.

## Limitations

- The dataset files and model outputs are downloaded/generated locally and are not committed with source code.
- The primary measured run uses one seed and 5% of the cleaned training split. It is not a multi-seed benchmark.
- The detected GPU is an RTX 4050 with 6 GiB VRAM; use short sequence lengths and small batches, and monitor for OOM.
- The measured test Macro-F1 is 0.5045; offensive-class F1 was 0.1909, so performance is uneven and needs more study.
- Fifteen live synthetic candidates across five generation calls have persisted (9 accepted pending human review and 6 rejected; one four-candidate call was caused by an unisolated test and is documented in the audit). Larger generation batches and A0/A4/A6 matched augmentation comparisons remain incomplete. The generator/validator API requires user-supplied credentials.
- Gujarati, Assamese, sentiment, NER, and cross-lingual transfer are scope interfaces only; they have no local task-specific data, trained checkpoints, or measured results.
- Independent human label adjudication, multiple seeds, three-model comparison, fully matched A0/A4/A6 results, and broader synopsis tasks remain incomplete.
- No result or model is claimed as successful solely because a configuration exists.

## Single-model training

The controlled trainer uses validation Macro-F1 for checkpoint selection. It reads test data only when `--evaluate-test` is explicitly passed. With limited disk, it saves model-only checkpoints by default; pass `--resumable-checkpoints` to save optimizer state as well.

```powershell
python scripts\train_model.py --model MuRIL --proportion 0.05 --epochs 3 --batch-size 2 --evaluate-test
```

Use `--model XLM-RoBERTa` or `--model "IndicBERT v2"` for those configured checkpoints. `--max-train-rows 64 --max-validation-rows 128` is a fast wiring smoke check, not a meaningful benchmark. The trainer records failures in `results/experiment_registry.csv` and writes successful metrics to `run.json`.


