from pathlib import Path

# ====== Path configuration ======
CONFIG_FILE = Path(__file__).resolve()
# ad_detection/joint_train/joint_config.py -> ad_detection/
PROJECT_ROOT = CONFIG_FILE.parent.parent

# ====== Audio ======
SAMPLING_RATE = 16000
JOINT_SECOND_LENGTH = 60        # 60秒音频
XLSR_MAX_TIME_STEPS = 50 * JOINT_SECOND_LENGTH  # ~3000 steps

# ====== Hardware (48GB) ======
PHASE1_BATCH_SIZE = 32           # Phase 1: no_grad, 显存低
PHASE1_GRAD_ACCUM = 1            # 有效batch = 32 * 1 = 32
PHASE2_BATCH_SIZE = 4            # Phase 2: 端到端反向传播
PHASE2_GRAD_ACCUM = 8            # 有效batch = 4 * 8 = 32
USE_AMP = True

# ====== Model freezing ======
XLSR_FINETUNE_LAST_N = 0         # XLSR 全冻结，梯度穿过回传到 FRCRN
USE_GRADIENT_CHECKPOINT = True    # 对frozen XLSR层用gradient checkpointing

# ====== AD classifier (和 baseline train/config.py 一致) ======
AD_DROPOUT = 0.2                 # baseline 用 0.2

# ====== Two-phase training ======
# Phase 1: 冻结 FRCRN, 只训练 AD 分类器 (让分类器在稳定特征上先学会)
# Phase 2: 解冻 FRCRN unet2, 联合训练 (分类器已收敛, 给 FRCRN 有意义的梯度)
WARMUP_EPOCHS = 20               # Phase 1 epoch 数
MAX_EPOCHS = 200                 # 总 epoch 数 (Phase 1 + Phase 2)

FRCRN_LR = 1e-5                  # Phase 2: FRCRN unet2
XLSR_LR = 1e-5                   # XLSR (当前未使用，XLSR全冻结)
AD_LR = 3e-3                     # AD分类器
WEIGHT_DECAY = 1e-2
ETA_MIN = 1e-6                   # CosineAnnealing最小学习率

# ====== Loss ======
# Phase 1: 只有 classify loss (ALPHA=0, FRCRN冻结)
# Phase 2: denoise loss ~0.003, classify loss ~0.67
ALPHA = 50.0                     # L_denoise 权重 (Phase 2 才生效)
BETA = 1.0                       # L_classify 权重

# ====== Early stopping ======
PATIENCE = 30                    # Phase 2 的 early stopping

# ====== Seeds ======
RANDOM_SEEDS = [21, 42, 84, 168, 336]
RANDOM_SEED = 42

# ====== Data loading ======
NUM_WORKERS = 4
TRAIN_SET_RATIO = 0.8
