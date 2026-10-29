from pathlib import Path
import time
import json
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_DIR = RESULTS_DIR / "model_benchmark"
MODEL_DIR = RESULTS_DIR / "models"
METRICS_PATH = BENCHMARK_DIR / "model_comparison_results.csv"
LABELS = ["hate", "offensive", "profane", "not"]
LABEL_DESCRIPTIONS = {
    "hate": "Hate speech",
    "offensive": "Offensive language",
    "profane": "Profanity",
    "not": "Not hate / neutral",
}
MODEL_ALIASES = {
    "MahaBERT": "l3cube-pune/marathi-bert",
    "MuRIL": "google/muril-base-cased",
    "XLM-RoBERTa": "FacebookAI/xlm-roberta-base",
    "IndicBERT v2": "ai4bharat/IndicBERTv2-MLM-only",
}

st.set_page_config(page_title="Marathi NLP Lab", page_icon="🪷", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --ink:#182230; --muted:#667085; --primary:#5b4bdb; }
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1440px; }
.hero { padding: 1.6rem 1.8rem; border-radius: 20px; background: linear-gradient(120deg,#211b55 0%,#5145cd 58%,#8d7cf6 100%); color:white; margin-bottom: 1.2rem; }
.hero h1 { color:white; font-size:2.2rem; margin:0 0 .4rem 0; letter-spacing:-.03em; }
.hero p { color:#eeeaff; font-size:1rem; margin:0; }
.eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; color:#d5ceff; margin-bottom:.5rem; }
.panel { border:1px solid #e7e7f0; border-radius:16px; padding:1rem 1.1rem; background:#fff; }
.small-muted { color:#667085; font-size:.9rem; }
.status-pill { display:inline-block; padding:.25rem .65rem; border-radius:999px; background:#f1efff; color:#4c3bc7; font-weight:700; font-size:.78rem; }
div[data-testid="stMetric"] { background:#fafaff; border:1px solid #e9e7fb; padding:1rem; border-radius:14px; }
</style>
""", unsafe_allow_html=True)


def get_checkpoint_dirs():
    found = []
    if MODEL_DIR.exists():
        for run_path in MODEL_DIR.rglob("run.json"):
            try:
                run = json.loads(run_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(run.get("status", "")).upper() != "COMPLETED":
                continue
            checkpoint = run_path.parent / "best"
            if (checkpoint / "config.json").exists() and (
                (checkpoint / "model.safetensors").exists() or (checkpoint / "pytorch_model.bin").exists()
            ):
                found.append(checkpoint)
    # Retain compatibility with checkpoints emitted by the previous notebook.
    if BENCHMARK_DIR.exists():
        for config in BENCHMARK_DIR.rglob("config.json"):
            checkpoint = config.parent
            if (checkpoint / "model.safetensors").exists() or (checkpoint / "pytorch_model.bin").exists():
                found.append(checkpoint)
    # Put the strongest locally trained run first in the classifier dropdown.
    # A tiny wiring smoke checkpoint is useful for debugging, but it should not
    # silently become the default prediction model when a benchmark is available.
    def checkpoint_rank(checkpoint):
        try:
            run_path = checkpoint.parent / "run.json"
            run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
            test_f1 = (run.get("test_metrics") or {}).get("macro_f1")
            validation_f1 = (run.get("validation_metrics") or {}).get("eval_macro_f1")
            # Checkpoint preference is based on validation data; the held-out
            # test score must not be used to choose a model for deployment.
            score = validation_f1 if validation_f1 is not None else test_f1
            return (float(score or 0), int(run.get("train_rows") or 0), str(checkpoint))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return (0.0, 0, str(checkpoint))

    return sorted(set(found), key=checkpoint_rank, reverse=True)


def load_metrics():
    if METRICS_PATH.exists():
        try:
            return pd.read_csv(METRICS_PATH)
        except Exception:
            pass
    # The unified trainer's run.json files are the source of truth when the
    # older notebook benchmark CSV has not been produced.
    records = []
    if MODEL_DIR.exists():
        for path in MODEL_DIR.rglob("run.json"):
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
                validation = run.get("validation_metrics") or {}
                test = run.get("test_metrics") or {}
                report = test.get("classification_report") or {}
                weighted = report.get("weighted avg", {})
                records.append({
                    "experiment_id": run.get("experiment_id", path.parent.name),
                    "model": run.get("model", "Unknown"),
                    "training_condition": "real_only",
                    "percentage": round(float(run.get("proportion", 0)) * 100),
                    "training_rows": run.get("train_rows"),
                    "validation_macro_f1": validation.get("eval_macro_f1"),
                    "test_accuracy": test.get("accuracy"),
                    "test_macro_f1": test.get("macro_f1"),
                    "test_weighted_f1": weighted.get("f1-score"),
                    "test_precision_macro": test.get("macro_precision"),
                    "test_recall_macro": test.get("macro_recall"),
                    "runtime_minutes": (float(run.get("training_seconds", 0)) / 60),
                    "status": run.get("status", "UNKNOWN").replace("COMPLETED", "RUN"),
                })
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
    return pd.DataFrame(records)


def pretty_label(label):
    return LABEL_DESCRIPTIONS.get(str(label).lower(), str(label))


def metric_value(df, col):
    if df.empty or col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.max()) if not vals.empty else None


def get_model_key(path):
    name = str(path).lower()
    if "muril" in name:
        return "MuRIL"
    if "mahabert" in name:
        return "MahaBERT"
    if "xlm" in name:
        return "XLM-RoBERTa"
    if "indicbert" in name:
        return "IndicBERT v2"
    return path.name


@st.cache_resource(show_spinner="Loading the selected classifier…")
def load_transformer(path):
    # Imported only when a user requests inference, keeping dashboard startup fast.
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(device)
    model.eval()
    return tokenizer, model, torch, device


with st.sidebar:
    st.markdown("## 🪷 Marathi NLP Lab")
    st.caption("Adaptive quality-aware synthetic data generation")
    page = st.radio("Navigate", ["Overview", "Classify Marathi Text", "Model Comparison", "Research Results", "About the Project"], label_visibility="collapsed")
    st.divider()
    checkpoints = get_checkpoint_dirs()
    if checkpoints:
        st.success(f"{len(checkpoints)} trained checkpoint(s) found")
    else:
        st.warning("No trained model checkpoints found yet")
    st.caption("Models: MahaBERT · MuRIL · XLM-RoBERTa · IndicBERT v2")

metrics = load_metrics()
checkpoints = get_checkpoint_dirs()

if page == "Overview":
    st.markdown("<div class='hero'><div class='eyebrow'>Research prototype · Marathi NLP</div><h1>Adaptive Marathi Text Intelligence</h1><p>Compare transformer models, classify Marathi text, and evaluate quality-aware synthetic data for low-resource hate-speech detection.</p></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classification labels", "4", help="Hate, Offensive, Profane, Not Hate")
    c2.metric("Candidate models", "4", help="MahaBERT, MuRIL, XLM-RoBERTa, IndicBERT v2")
    c3.metric("Saved checkpoints", str(len(checkpoints)))
    c4.metric("Benchmark rows", str(len(metrics)))
    st.subheader("Project workflow")
    cols = st.columns(4)
    steps = [("01", "Dataset", "L3Cube-MahaHate and consistent train/validation/test splits."), ("02", "Generate", "Create candidate Marathi examples using the configured LLM."), ("03", "Quality filter", "Score candidates with SDQS signals, duplicate checks, and quality rules."), ("04", "Evaluate", "Compare models using Macro-F1, accuracy, precision, and recall.")]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"<div class='panel'><span class='status-pill'>STEP {num}</span><h3>{title}</h3><p class='small-muted'>{desc}</p></div>", unsafe_allow_html=True)
    st.subheader("Quick start")
    st.markdown("1. Put the official MahaHate train, validation, and test files in `data/maha_hate`.\n2. Train a model with `python scripts/train_model.py --model MuRIL --proportion 0.05 --epochs 3 --batch-size 2 --evaluate-test`.\n3. Use **Classify Marathi Text** to run inference and **Model Comparison** to inspect the saved metrics.")
    if not checkpoints:
        st.info("The interface is ready, but predictions require fine-tuned checkpoints. Run the benchmark section in the notebook first; this dashboard does not invent sample metrics or model predictions.")

elif page == "Classify Marathi Text":
    st.title("Classify Marathi Text")
    st.write("Enter Marathi text, Romanized Marathi, or Marathi-English code-mixed text. The selected fine-tuned model returns a predicted class and confidence scores.")
    if not checkpoints:
        st.warning("No fine-tuned checkpoint was found. Train a classifier with scripts\\train_model.py, then refresh this page.")
        st.code("python -m pip install -r requirements.txt\nstreamlit run app.py", language="bash")
    else:
        path_by_name = {f"{get_model_key(p)} — {p.parent.name}": p for p in checkpoints}
        chosen = st.selectbox("Choose trained model", list(path_by_name.keys()))
        sentence = st.text_area("Text to classify", placeholder="उदा. हा मजकूर येथे लिहा…", height=140, max_chars=2000)
        predict = st.button("Analyze text", type="primary", use_container_width=False)
        if predict:
            if not sentence.strip():
                st.error("Please enter a sentence first.")
            else:
                with st.spinner("Loading model and analyzing text…"):
                    try:
                        path = path_by_name[chosen]
                        tokenizer, model, torch, device = load_transformer(path)
                        batch = tokenizer(sentence.strip(), return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
                        with torch.no_grad():
                            probs = torch.softmax(model(**batch).logits, dim=-1)[0].detach().cpu().tolist()
                        config = model.config
                        id2label = getattr(config, "id2label", {}) or {}
                        names = [str(id2label.get(i, id2label.get(str(i), LABELS[i]))).lower() for i in range(len(probs))]
                        # Prefer the project's canonical label ordering where the saved model uses generic LABEL_i names.
                        if all(n.startswith("label_") for n in names) or len(names) != 4:
                            names = LABELS[:len(probs)]
                        best_idx = max(range(len(probs)), key=lambda i: probs[i])
                        pred = names[best_idx]
                        st.success("Analysis complete")
                        a, b = st.columns([1, 1])
                        with a:
                            st.markdown("### Predicted class")
                            st.subheader(pretty_label(pred))
                            st.metric("Model confidence", f"{probs[best_idx]*100:.1f}%")
                            st.caption("Confidence is the model's probability score, not a guarantee that the label is correct.")
                        with b:
                            st.markdown("### Class probability")
                            prob_df = pd.DataFrame({"Class": [pretty_label(n) for n in names], "Probability": probs})
                            st.plotly_chart(px.bar(prob_df, x="Probability", y="Class", orientation="h", range_x=[0, 1], text=prob_df["Probability"].map(lambda x: f"{x:.1%}")), use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not load or run this checkpoint: {e}")
                        st.caption("Check that the checkpoint folder contains tokenizer and model files and that PyTorch/Transformers are installed.")

elif page == "Model Comparison":
    st.title("Model Comparison")
    st.write("Compare the actual results recorded by the notebook. Macro-F1 should be the primary model-selection metric for the four-class task.")
    if metrics.empty:
        st.warning("No benchmark results are available yet. Run the notebook benchmark first. Results will appear here automatically after the CSV is created.")
        st.code("results/model_benchmark/model_comparison_results.csv", language="text")
    else:
        df = metrics.copy()
        if "status" in df.columns:
            df = df[df.status.astype(str).str.upper().eq("RUN")]
        if df.empty:
            st.warning("The benchmark CSV exists, but no runs have status RUN yet. Inspect the error column and rerun failed models.")
        else:
            for col in ["validation_macro_f1", "test_macro_f1", "test_accuracy", "test_weighted_f1", "test_precision_macro", "test_recall_macro"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            model_filter = st.multiselect("Filter models", sorted(df.model.dropna().unique()), default=sorted(df.model.dropna().unique()) if "model" in df else [])
            if model_filter and "model" in df:
                df = df[df.model.isin(model_filter)]
            best = df.sort_values("validation_macro_f1", ascending=False).iloc[0] if "validation_macro_f1" in df and df.validation_macro_f1.notna().any() else None
            if best is not None:
                st.markdown(f"<div class='panel'><span class='status-pill'>VALIDATION-SELECTED MODEL</span><h2>{best.get('model','Unknown')}</h2><p class='small-muted'>Training condition: {best.get('training_condition','Unknown')} · Validation Macro-F1: {best['validation_macro_f1']:.4f}</p></div>", unsafe_allow_html=True)
            score_cols = [c for c in ["model", "training_condition", "percentage", "training_rows", "validation_macro_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1", "test_precision_macro", "test_recall_macro", "runtime_minutes", "status"] if c in df.columns]
            st.dataframe(df[score_cols].sort_values("test_macro_f1", ascending=False) if "test_macro_f1" in df.columns else df[score_cols], use_container_width=True, hide_index=True)
            if "test_macro_f1" in df.columns and df.test_macro_f1.notna().any():
                plotdf = df.dropna(subset=["test_macro_f1"]).copy()
                plotdf["Series"] = plotdf["model"].astype(str) + " · " + plotdf.get("training_condition", pd.Series("", index=plotdf.index)).astype(str)
                st.plotly_chart(px.bar(plotdf, x="Series", y="test_macro_f1", color="model", text=plotdf["test_macro_f1"].map(lambda x: f"{x:.3f}"), title="Held-out test Macro-F1 by model and training condition"), use_container_width=True)
            st.caption("Choose the production model using validation Macro-F1. Test scores are for final reporting only and should not be used repeatedly to tune the model.")

elif page == "Research Results":
    st.title("Research Results")
    st.write("This page summarizes benchmark evidence and the status of the synthetic-data experiment. It only displays recorded outputs.")
    if metrics.empty:
        st.info("No results file detected. Complete the benchmark run to populate this page with measured scores.")
    else:
        successful = metrics[metrics.get("status", pd.Series(index=metrics.index, dtype=str)).astype(str).str.upper().eq("RUN")].copy()
        if successful.empty:
            st.warning("No successful benchmark rows are available yet.")
        else:
            best_val = successful.sort_values("validation_macro_f1", ascending=False).iloc[0] if "validation_macro_f1" in successful else None
            best_test = successful.sort_values("test_macro_f1", ascending=False).iloc[0] if "test_macro_f1" in successful else None
            x, y = st.columns(2)
            with x:
                st.markdown("#### Recommended model")
                if best_val is not None:
                    st.metric(str(best_val.get("model", "Unknown")), f"{float(best_val.get('validation_macro_f1', float('nan'))):.3f}", "Validation Macro-F1")
                    st.caption("Selection uses validation data.")
            with y:
                st.markdown("#### Highest observed test score")
                if best_test is not None:
                    st.metric(str(best_test.get("model", "Unknown")), f"{float(best_test.get('test_macro_f1', float('nan'))):.3f}", "Test Macro-F1")
                    st.caption("Descriptive only; not the model-selection rule.")
            st.dataframe(successful, use_container_width=True, hide_index=True)
            st.download_button("Download benchmark results CSV", successful.to_csv(index=False).encode("utf-8"), file_name="model_comparison_results.csv", mime="text/csv")
    st.divider()
    st.subheader("What counts as a successful research result?")
    st.markdown("- All three models complete training and evaluation on identical data splits.\n- Validation Macro-F1 determines the selected model.\n- Held-out test metrics are reported once for final comparison.\n- A0 (real-only) is compared with A4 (SDQS) and A6 (adaptive) where those runs actually completed.\n- Any missing experiment, failed run, or empty adaptive pool is shown as missing/inconclusive—not treated as an improvement.")

else:
    st.title("About the Project")
    st.markdown("""
### Adaptive Quality-Aware Synthetic Data Generation for Low-Resource Marathi NLP

**Goal:** investigate whether carefully filtered and adaptively generated synthetic Marathi text can improve four-class hate-speech classification when only a small portion of real training data is available.

**Dataset:** L3Cube-MahaHate (four labels: hate, offensive, profane, not hate).

**Classifier comparison:** MuRIL, XLM-RoBERTa, and IndicBERT v2.

**Synthetic-data pipeline:** candidate generation, SDQS-based quality scoring, duplicate/similarity checks, and optional difficulty/adaptive targeting.

**Main metric:** Macro-F1, because it gives each class equal weight. Accuracy, weighted-F1, precision, recall, per-class performance, and runtime provide supporting evidence.

**Research caution:** a frontend is a demonstration and reporting layer, not evidence of better model performance by itself. Claims should be based on successful, reproducible benchmark runs and a held-out test set.
""")

st.divider()
st.caption("Marathi NLP Lab · Research prototype. Model predictions may be wrong; do not use this tool to make consequential decisions about individuals.")
