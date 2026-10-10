"""
config.py
---------
Central configuration for the Sign Language Recognition System.

Keeping every tunable value in one place is what lets the rest of the
codebase stay maintainable: paths, thresholds and hyperparameters are never
hard-coded inside the logic that uses them.
"""

import os

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

LANDMARKS_CSV = os.path.join(DATA_PROCESSED_DIR, "landmarks_dataset.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "sign_classifier.pkl")
LABEL_ENCODER_FILE = os.path.join(MODELS_DIR, "label_encoder.pkl")
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")

# ---------------------------------------------------------------------------
# MediaPipe Hands configuration
# ---------------------------------------------------------------------------
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5
NUM_LANDMARKS = 21          # MediaPipe returns 21 (x, y, z) points per hand
FEATURES_PER_LANDMARK = 3
FEATURE_VECTOR_LENGTH = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 63

# ---------------------------------------------------------------------------
# Training / model configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_MODEL_TYPE = "random_forest"   # random_forest | svm | mlp
N_ESTIMATORS = 200
CROSS_VAL_FOLDS = 5

# ---------------------------------------------------------------------------
# Runtime / non-functional configuration
# ---------------------------------------------------------------------------
LOGGING_LEVEL = "INFO"
FRAME_SKIP = 1                  # process every Nth frame from a video (perf)
MAX_VIDEO_FRAMES = 5000          # safety cap to bound resource usage
CONFIDENCE_THRESHOLD = 0.55      # below this, prediction is reported as "Unknown"

# Default class labels for the demo dataset (A-E static ASL alphabet signs).
# A real deployment simply points collect_data.py at more classes.
DEFAULT_CLASSES = ["A", "B", "C", "D", "E"]


def ensure_directories():
    """Create all working directories if they do not already exist."""
    for d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)
