from pathlib import Path

# ====== Path configuration ======
# Get the absolute path of the config.py file
CONFIG_FILE = Path(__file__).resolve()
# ad_detection/train/utils/config.py -> ad_detection/
PROJECT_ROOT = CONFIG_FILE.parent.parent.parent

# ====== Training parameters ======
MAX_EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 3e-3         # Best learning rate is 3e-3
ETA_MIN = 1e-6               # Cosine annealing minimum learning rate
WEIGHT_DECAY = 1e-2
TRAIN_SET_RATTIO = 0.8
PATIENCE = 10                # Early stopping patience (epochs without val_acc improvement)

# ====== XLSR model parameters ======
XLSR_DROPOUT = 0.2

# ====== XLSR features extraction parameters ======
SECOND_LENGTH = 60     # XLSR extraction audio length (seconds)
XLSR_MAX_TIME_STEPS = 50 * SECOND_LENGTH  # XLSR time steps (approximately 50 steps/second, downsampling ratio ~320)

# ====== eGeMAPS features extraction parameters ======
FEAT_SEQ_LEN = 10      # Number of audio segments when extracting eGeMaps

# ====== eGeMAPS model parameters ======
EGEMAPS_DROPOUT = 0.3

# ====== Data loading parameters ======
NUM_WORKERS = 4
RANDOM_SEED = 42
SAMPLING_RATE = 16000  # Audio sampling rate