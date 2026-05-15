# ====== SLS feature extraction (reuses frozen XLS-R backbone) ======
SECOND_LENGTH = 60
MAX_TIME_STEPS = 50 * SECOND_LENGTH

# ====== Model ======
DROPOUT = 0.2
PRE_POOL_KERNEL = 4

# ====== Training ======
LEARNING_RATE = 3e-4
ETA_MIN = 1e-6
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 60
PATIENCE = 5
