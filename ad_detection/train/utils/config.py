from pathlib import Path

# ====== Path configuration ======
# Get the absolute path of the config.py file
CONFIG_FILE = Path(__file__).resolve()
# ad_detection/train/utils/config.py -> ad_detection/
PROJECT_ROOT = CONFIG_FILE.parent.parent.parent

# ====== Training parameters ======
MAX_EPOCHS = 60              
BATCH_SIZE = 32              
LEARNING_RATE = 3e-3         #Best learning rate is 3e-3
ETA_MIN = 1e-6               # Cosine annealing minimum learning rate
WARMUP_STEPS = 100
WEIGHT_DECAY = 1e-2  # Reduced for simpler model
TRAIN_SET_RATTIO = 0.8

# ====== Model parameters ======
# XLSR_DIM_HIDDEN = 32 perform well
XLSR_DIM_INPUT = 1024               # XLSR feature dimension (output from XLSR-53 model)
XLSR_DIM_HIDDEN = 26                # XLSR Network Hidden dimension
XLSR_DROPOUT = 0.2                  # XLSR Dropout ratio
RANDOM_SEEDS = [21, 42, 84, 168, 336]

# ====== XLSR features extraction parameters ======
SECOND_LENGTH = 60     # XLSR extraction audio length (seconds)
XLSR_MAX_TIME_STEPS = 50 * SECOND_LENGTH  # XLSR time steps (approximately 50 steps/second, downsampling ratio ~320)

# ====== Data loading parameters ======
NUM_WORKERS = 4
RANDOM_SEED = 42
SAMPLING_RATE = 16000  # Audio sampling rate