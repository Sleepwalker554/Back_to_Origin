# ====== XLSR feature extraction ======
SECOND_LENGTH = 60
MAX_TIME_STEPS = 50 * SECOND_LENGTH

# ====== Model ======
DROPOUT = 0.2

# ====== Training ======
BATCH_SIZE = 16
LEARNING_RATE = 3e-3
ETA_MIN = 1e-6
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 60
PATIENCE = 5
