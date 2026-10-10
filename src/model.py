"""
model.py
--------
Model factory and persistence helpers. Isolating model construction here
means train_model.py, evaluate.py and predict.py all build/load models
the exact same way, and swapping algorithms is a one-line change instead
of a scattered refactor.
"""

import json
import pickle
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def build_model(model_type: str = config.DEFAULT_MODEL_TYPE):
    """Factory returning an untrained scikit-learn estimator."""
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    if model_type == "svm":
        return SVC(kernel="rbf", probability=True, random_state=config.RANDOM_STATE)
    if model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            random_state=config.RANDOM_STATE,
        )
    raise ValueError(f"Unknown model_type '{model_type}'. "
                      "Choose one of: random_forest, svm, mlp.")


def save_model(model: Any, encoder: Any, path: str = config.MODEL_FILE,
               encoder_path: str = config.LABEL_ENCODER_FILE) -> None:
    config.ensure_directories()
    with open(path, "wb") as f:
        pickle.dump(model, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)
    logger.info("Saved model to %s and label encoder to %s", path, encoder_path)


def load_model(path: str = config.MODEL_FILE, encoder_path: str = config.LABEL_ENCODER_FILE):
    with open(path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


def save_metrics(metrics: dict, path: str = config.METRICS_FILE) -> None:
    config.ensure_directories()
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", path)
