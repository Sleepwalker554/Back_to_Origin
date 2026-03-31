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
JOINT_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32  # 有效batch = 32
USE_AMP = True

# ====== Model freezing ======
XLSR_FINETUNE_LAST_N = 0         # XLSR 全冻结，梯度穿过回传到 FRCRN
USE_GRADIENT_CHECKPOINT = True    # 对frozen XLSR层用gradient checkpointing

# ====== Training ======
MAX_EPOCHS = 50
FRCRN_LR = 1e-5                  # FRCRN unet2
XLSR_LR = 1e-5                   # XLSR (当前未使用，XLSR全冻结)
AD_LR = 3e-3                     # AD分类器
WEIGHT_DECAY = 1e-2
ETA_MIN = 1e-6                   # CosineAnnealing最小学习率

# ====== Loss ======
# denoise loss ~0.003, classify loss ~0.67, 放大 ALPHA 让两个 loss 同量级
ALPHA = 50.0                    # L_denoise 权重 (0.003 * 50 ≈ 0.15)
BETA = 1.0                       # L_classify 权重

# ====== Early stopping ======
PATIENCE = 10

# ====== Seeds ======
RANDOM_SEEDS = [21, 42, 84, 168, 336]
RANDOM_SEED = 42

# ====== Data loading ======
NUM_WORKERS = 4
TRAIN_SET_RATIO = 0.8
