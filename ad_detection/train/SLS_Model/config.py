# ====== SLS feature extraction (reuses frozen XLS-R backbone) ======
SECOND_LENGTH = 60
MAX_TIME_STEPS = 50 * SECOND_LENGTH  # 3000

# ====== Model ======
DROPOUT = 0.2
# AvgPool1d kernel/stride before Conv1d. T=3000 -> T // PRE_POOL_KERNEL.
# 4 = 80ms/frame (recommended), 2 = 40ms (less smoothing), 8 = 160ms (more).
PRE_POOL_KERNEL = 4

# ====== Training ======
# Note: SLS batch size is set per-notebook (SLS_BATCH_SIZE) since SLS cached
# features are (L=24, T, 1024) — ~24x larger per sample than XLSR — and the
# right batch size depends on the dataset size and host RAM.
LEARNING_RATE = 3e-4
ETA_MIN = 1e-6
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 60
PATIENCE = 5
