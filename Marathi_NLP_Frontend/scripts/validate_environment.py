from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
modules = ("torch", "transformers", "datasets", "pandas", "sklearn", "streamlit", "plotly", "sentence_transformers", "groq")
report = {"python": sys.version, "platform": platform.platform(), "modules": {name: importlib.util.find_spec(name) is not None for name in modules}, "dataset_directory_present": (ROOT / "data").exists()}
if report["modules"]["torch"]:
    import torch
    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
if shutil.which("nvidia-smi"):
    proc = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], capture_output=True, text=True)
    report["gpu"] = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
print(json.dumps(report, indent=2))
