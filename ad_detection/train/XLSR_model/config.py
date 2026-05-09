# ====== XLSR feature extraction ======
SECOND_LENGTH = 60                # extraction audio length (seconds)
MAX_TIME_STEPS = 50 * SECOND_LENGTH  # ~50 frames/s after 320x downsampling -> 3000

# ====== Model ======
DROPOUT = 0.2

# ====== Training ======
BATCH_SIZE = 16
LEARNING_RATE = 3e-3
ETA_MIN = 1e-6                    # cosine annealing minimum lr
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 60
PATIENCE = 10                     # early-stopping patience (epochs without val_acc improvement)
