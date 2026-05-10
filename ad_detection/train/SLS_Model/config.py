# ====== SLS feature extraction (reuses frozen XLS-R backbone) ======
SECOND_LENGTH = 60
MAX_TIME_STEPS = 50 * SECOND_LENGTH  # 3000

# ====== Model ======
DROPOUT = 0.2

# ====== Training ======
# SLS cached features are (L=24, T, 1024) — ~24x larger per sample than XLSR.
# Keep BATCH_SIZE small to fit GPU activation memory.
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
ETA_MIN = 1e-6
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 60
PATIENCE = 10
