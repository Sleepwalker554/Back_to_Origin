from pathlib import Path

# ====== Path configuration ======
# ad_detection/train/utils/config.py -> ad_detection/
CONFIG_FILE = Path(__file__).resolve()
PROJECT_ROOT = CONFIG_FILE.parent.parent.parent

# ====== Cross-model constants ======
SAMPLING_RATE = 16000
NUM_WORKERS = 4

# ====== Data split ======
RANDOM_SEED = 42
TRAIN_SET_RATTIO = 0.8
