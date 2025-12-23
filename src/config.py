# src/config.py
import os
from pathlib import Path

# Base project directory (root of the repo)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "Alzheimers disease MRI images"


# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [DATA_DIR, RAW_DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Training hyperparameters
NUM_CLASSES = 3  # NC, MCI, AD
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_EPOCHS = 35
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

# Train/val/test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Checkpoint file
BEST_MODEL_PATH = CHECKPOINT_DIR / "alznet_best.pth"
