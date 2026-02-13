from pathlib import Path

# ====== Path configuration ======
# Get the absolute path of the config.py file
CONFIG_FILE = Path(__file__).resolve()
# ad_detection/train/config.py -> ad_detection/
PROJECT_ROOT = CONFIG_FILE.parent.parent

# ====== Training parameters ======
MAX_EPOCHS = 60              
BATCH_SIZE = 32              
LEARNING_RATE = 3e-3         #Best learning rate is 3e-3
WARMUP_STEPS = 100
WEIGHT_DECAY = 1e-2  # Reduced for simpler model
TRAIN_SET_RATTIO = 0.8

# ====== Model parameters ======
# XLSR_DIM_HIDDEN = 32 perform well
EGEMAPS_DIM_INPUT = 25
XLSR_DIM_INPUT = 1024               # XLSR feature dimension (output from XLSR-53 model)
EGEMAPS_DIM_HIDDEN = 14             # eGeMaps Network Hidden dimension
XLSR_DIM_HIDDEN = 26                # XLSR Network Hidden dimension
XLSR_DROPOUT = 0.2                  # XLSR Dropout ratio
EGEMAPS_DROPOUT = 0.3               # eGeMaps Dropout ratio
RANDOM_SEEDS = [21, 42, 84, 168, 336]

# ====== DANN (Domain Adversarial) parameters ======
DANN_LAMBDA_CLASS = 1.0             # Classification loss weight
DANN_LAMBDA_DOMAIN = 0.1            # Domain adversarial loss weight (reduced to prevent over-alignment)

# ====== eGeMAPS features extraction parameters ======
FEAT_SEQ_LEN = 10      # Number of audio segments when extracting eGeMaps

# ====== XLSR features extraction parameters ======
SECOND_LENGTH = 60     # XLSR extraction audio length (seconds)
XLSR_MAX_TIME_STEPS = 50 * SECOND_LENGTH  # XLSR time steps (approximately 50 steps/second, downsampling ratio ~320)

# ====== Data loading parameters ======
NUM_WORKERS = 4
RANDOM_SEED = 42
SAMPLING_RATE = 16000  # Audio sampling rate