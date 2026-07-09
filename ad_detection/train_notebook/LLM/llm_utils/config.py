import os
from pathlib import Path

# ====== Paths ======
CONFIG_FILE = Path(__file__).resolve()
PROJECT_ROOT = CONFIG_FILE.parents[3]
CACHE_DIR = os.environ.get("LLM_MODEL_CACHE", "/root/autodl-tmp/LLM_Model")

# ====== HuggingFace env (set at import time) ======
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", CACHE_DIR)
