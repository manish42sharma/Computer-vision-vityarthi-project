"""
evaluate.py
-----------
Functional Module 2b: Model Evaluation & Reporting.

Loads the trained model, re-runs it against a held-out test split (or
an independently supplied CSV), and prints/exports a full classification
report and confusion matrix. Separated from train_model.py so a model
can be re-evaluated at any time without retraining.
"""

import argparse
import json
import os

from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.dataset import load_dataset, split_features_labels, train_test_split_dataset
from src.model import load_model
from src.utils import get_logger, safe_run, validate_path_exists

logger = get_logger(__name__)


def evaluate(csv_path: str, use_held_out_split: bool, output_json: str = None):
    validate_path_exists(config.MODEL_FILE, "Trained model")
    model, encoder = load_model()

    df = load_dataset(csv_path)
    X, y, encoder_check = split_features_labels(df)

    if use_held_out_split:
        _, X_eval, _, y_eval = train_test_split_dataset(X, y)
    else:
        X_eval, y_eval = X, y

    y_pred = model.predict(X_eval)

    report = classification_report(
        y_eval, y_pred, target_names=[str(c) for c in encoder.classes_],
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_eval, y_pred).tolist()

    result = {
        "n_samples_evaluated": len(y_eval),
        "classes": list(map(str, encoder.classes_)),
        "classification_report": report,
        "confusion_matrix": cm,
    }

    if output_json:
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Wrote evaluation report to %s", output_json)

    return result


def _print_report(result: dict):
    print("\n=== Evaluation Report ===")
    print(f"Samples evaluated: {result['n_samples_evaluated']}")
    print(f"Classes: {', '.join(result['classes'])}\n")

    report = result["classification_report"]
    print(f"{'class':<10}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
    for label, stats in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        print(f"{label:<10}{stats['precision']:>10.3f}{stats['recall']:>10.3f}"
              f"{stats['f1-score']:>10.3f}{stats['support']:>10.0f}")
    print(f"\nOverall accuracy: {report['accuracy']:.3f}")
    print(f"Macro F1-score  : {report['macro avg']['f1-score']:.3f}")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    for row in result["confusion_matrix"]:
        print(" ".join(f"{v:4d}" for v in row))


@safe_run(logger)
def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained sign classifier.")
    parser.add_argument("--csv", default=config.LANDMARKS_CSV,
                         help="Dataset CSV to evaluate against.")
    parser.add_argument("--full-dataset", action="store_true",
                         help="Evaluate on the ENTIRE csv instead of a fresh held-out split.")
    parser.add_argument("--output-json", default=os.path.join(config.MODELS_DIR, "evaluation_report.json"))
    args = parser.parse_args()

    result = evaluate(args.csv, use_held_out_split=not args.full_dataset,
                       output_json=args.output_json)
    _print_report(result)


if __name__ == "__main__":
    main()
