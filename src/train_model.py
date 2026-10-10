"""
train_model.py
---------------
Functional Module 2: Model Training.

Trains a classifier on the extracted hand-landmark dataset and persists
it to models/sign_classifier.pkl. Fully command-line driven; prints a
validation accuracy summary on completion so the user gets immediate
feedback with no GUI needed.
"""

import argparse
import time

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from src import config
from src.dataset import load_dataset, split_features_labels, train_test_split_dataset
from src.model import build_model, save_model, save_metrics
from src.utils import get_logger, safe_run, timeit

logger = get_logger(__name__)


@timeit(logger)
def train(model_type: str, csv_path: str, cv_folds: int):
    df = load_dataset(csv_path)
    X, y, encoder = split_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y)

    logger.info("Training '%s' on %d samples (%d train / %d test)...",
                model_type, len(X), len(X_train), len(X_test))

    model = build_model(model_type)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    cv_scores = None
    if cv_folds and cv_folds > 1 and len(X_train) >= cv_folds:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)

    metrics = {
        "model_type": model_type,
        "n_samples": len(X),
        "n_classes": len(encoder.classes_),
        "classes": list(encoder.classes_),
        "train_accuracy": round(float(train_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "training_time_seconds": round(train_seconds, 4),
        "cross_val_mean_accuracy": round(float(cv_scores.mean()), 4) if cv_scores is not None else None,
        "cross_val_std": round(float(cv_scores.std()), 4) if cv_scores is not None else None,
    }

    save_model(model, encoder)
    save_metrics(metrics)
    return metrics


@safe_run(logger)
def main():
    parser = argparse.ArgumentParser(
        description="Train the sign-language landmark classifier."
    )
    parser.add_argument("--model-type", choices=["random_forest", "svm", "mlp"],
                         default=config.DEFAULT_MODEL_TYPE)
    parser.add_argument("--csv", default=config.LANDMARKS_CSV,
                         help="Path to the processed landmarks dataset CSV.")
    parser.add_argument("--cv-folds", type=int, default=config.CROSS_VAL_FOLDS,
                         help="Number of cross-validation folds (0 to disable).")
    args = parser.parse_args()

    metrics = train(args.model_type, args.csv, args.cv_folds)

    print("\n=== Training Summary ===")
    for k, v in metrics.items():
        print(f"{k:26s}: {v}")
    print(f"\nModel saved to: {config.MODEL_FILE}")


if __name__ == "__main__":
    main()
