"""
test_pipeline.py
-----------------
Validation tests covering every functional module:
  - feature extraction (landmark -> vector normalisation)
  - dataset loading / splitting
  - model training, persistence and inference
  - CLI-level error handling for bad inputs

Run with:  pytest tests/ -v
"""

import os
import types

import numpy as np
import pandas as pd
import pytest

from src import config
from src.feature_extraction import HandLandmarkExtractor
from src.dataset import split_features_labels, train_test_split_dataset
from src.model import build_model, save_model, load_model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class _FakeLandmark:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _FakeHandLandmarks:
    def __init__(self, points):
        self.landmark = [_FakeLandmark(*p) for p in points]


def test_landmarks_to_vector_shape_and_translation_invariance():
    points = [(0.5 + i * 0.01, 0.5 + i * 0.01, 0.0) for i in range(21)]
    fake = _FakeHandLandmarks(points)

    vector = HandLandmarkExtractor._landmarks_to_vector(fake)

    assert vector.shape == (config.FEATURE_VECTOR_LENGTH,)
    # Wrist (landmark 0) must be at the origin after translation-normalisation.
    assert np.allclose(vector[0:3], [0.0, 0.0, 0.0], atol=1e-6)


def test_feature_names_length_matches_vector():
    assert len(HandLandmarkExtractor.feature_names()) == config.FEATURE_VECTOR_LENGTH


def test_extract_on_blank_frame_returns_none():
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    with HandLandmarkExtractor(static_image_mode=True) as extractor:
        vector, landmarks = extractor.extract(blank)
    assert vector is None
    assert landmarks is None


def test_extract_on_empty_frame_is_handled_gracefully():
    with HandLandmarkExtractor(static_image_mode=True) as extractor:
        vector, landmarks = extractor.extract(np.array([]))
    assert vector is None


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------

def _make_synthetic_df(n_per_class=30, n_classes=3):
    rng = np.random.RandomState(0)
    rows = []
    for c in range(n_classes):
        center = rng.uniform(-1, 1, size=config.FEATURE_VECTOR_LENGTH)
        for _ in range(n_per_class):
            sample = center + rng.normal(0, 0.05, size=config.FEATURE_VECTOR_LENGTH)
            rows.append(list(sample) + [f"class_{c}"])
    cols = HandLandmarkExtractor.feature_names() + ["label"]
    return pd.DataFrame(rows, columns=cols)


def test_split_features_labels_shapes():
    df = _make_synthetic_df()
    X, y, encoder = split_features_labels(df)
    assert X.shape == (len(df), config.FEATURE_VECTOR_LENGTH)
    assert y.shape == (len(df),)
    assert len(encoder.classes_) == 3


def test_train_test_split_is_stratified_and_covers_all_classes():
    df = _make_synthetic_df()
    X, y, encoder = split_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y, test_size=0.25)

    assert len(X_train) + len(X_test) == len(X)
    assert set(np.unique(y_train)) == set(np.unique(y))


# ---------------------------------------------------------------------------
# Model train / save / load / predict round-trip
# ---------------------------------------------------------------------------

def test_model_training_and_roundtrip_prediction(tmp_path):
    df = _make_synthetic_df(n_per_class=40)
    X, y, encoder = split_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y)

    model = build_model("random_forest")
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    assert accuracy > 0.8  # well-separated synthetic clusters should be easy

    model_path = tmp_path / "model.pkl"
    encoder_path = tmp_path / "encoder.pkl"
    save_model(model, encoder, str(model_path), str(encoder_path))

    loaded_model, loaded_encoder = load_model(str(model_path), str(encoder_path))
    loaded_preds = loaded_model.predict(X_test)
    original_preds = model.predict(X_test)
    assert np.array_equal(loaded_preds, original_preds)
    assert list(loaded_encoder.classes_) == list(encoder.classes_)


@pytest.mark.parametrize("model_type", ["random_forest", "svm", "mlp"])
def test_build_model_all_supported_types(model_type):
    model = build_model(model_type)
    assert model is not None


def test_build_model_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_model("not_a_real_model")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_load_dataset_missing_file_raises():
    from src.dataset import load_dataset
    with pytest.raises(FileNotFoundError):
        load_dataset("/nonexistent/path/dataset.csv")
