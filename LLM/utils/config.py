import os
from pathlib import Path

# ====== Paths ======
PROJECT_ROOT = Path("/root/autodl-tmp/Back_to_Origin/ad_detection")
CACHE_DIR    = "/root/autodl-tmp/LLM_Model"

# ====== HuggingFace env (set at import time) ======
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", CACHE_DIR)
