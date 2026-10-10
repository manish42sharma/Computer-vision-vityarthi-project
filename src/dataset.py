"""
dataset.py
----------
Loads the processed landmark CSV, validates it, and produces
train/test splits. Kept separate from train_model.py so the same
loading logic can be reused by evaluate.py and by unit tests.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src import config
from src.utils import get_logger, validate_non_empty, validate_path_exists

logger = get_logger(__name__)


def load_dataset(csv_path: str = config.LANDMARKS_CSV) -> pd.DataFrame:
    validate_path_exists(csv_path, "Landmarks dataset CSV")
    df = pd.read_csv(csv_path)

    expected_cols = config.FEATURE_VECTOR_LENGTH + 1  # + label
    if df.shape[1] != expected_cols:
        raise ValueError(
            f"Dataset at {csv_path} has {df.shape[1]} columns; "
            f"expected {expected_cols} (63 landmark features + label)."
        )
    validate_non_empty(df, "Loaded dataset")
    logger.info("Loaded dataset: %d samples across %d classes",
                len(df), df["label"].nunique())
    return df


def split_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"].values)

    return X, y, encoder


def train_test_split_dataset(
    X: np.ndarray, y: np.ndarray,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
):
    # stratify keeps class proportions balanced across the split, which
    # matters a lot on a small, hand-collected dataset.
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
