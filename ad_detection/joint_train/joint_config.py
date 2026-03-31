from pathlib import Path

# ====== Path configuration ======
CONFIG_FILE = Path(__file__).resolve()
# ad_detection/joint_train/joint_config.py -> ad_detection/
PROJECT_ROOT = CONFIG_FILE.parent.parent

# ====== Audio ======
SAMPLING_RATE = 16000
JOINT_SECOND_LENGTH = 60        # 60秒音频
XLSR_MAX_TIME_STEPS = 50 * JOINT_SECOND_LENGTH  # ~3000 steps

# ====== Hardware (RTX 5090 32GB) ======
JOINT_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16  # 有效batch = 32
USE_AMP = True

# ====== Model freezing ======
XLSR_FINETUNE_LAST_N = 3         # 解冻XLSR最后3层transformer
USE_GRADIENT_CHECKPOINT = True    # 对frozen XLSR层用gradient checkpointing

# ====== Training ======
MAX_EPOCHS = 50
FRCRN_LR = 1e-5                  # FRCRN unet2
XLSR_LR = 1e-5                   # XLSR 后3层
AD_LR = 3e-3                     # AD分类器
WEIGHT_DECAY = 1e-2
ETA_MIN = 1e-6                   # CosineAnnealing最小学习率

# ====== Loss ======
ALPHA = 1.0                      # L_denoise 权重
BETA = 1.0                       # L_classify 权重

# ====== Early stopping ======
PATIENCE = 10

# ====== Seeds ======
RANDOM_SEEDS = [21, 42, 84, 168, 336]
RANDOM_SEED = 42

# ====== Data loading ======
NUM_WORKERS = 4
TRAIN_SET_RATIO = 0.8
