"""
generate_demo_dataset.py
-------------------------
Utility module: generates a small SYNTHETIC landmark dataset so the
rest of the pipeline (train_model.py, evaluate.py, predict.py) can be
exercised end-to-end from the command line before a user has collected
any real hand-landmark data with collect_data.py.

This is a convenience/testing tool, NOT a substitute for real data.
It draws each class from its own fixed random Gaussian "shape" in the
63-dimensional landmark-feature space, which is enough to sanity-check
the whole pipeline (I/O, training, evaluation, persistence, inference)
without needing a webcam or a labelled image dataset yet.

Usage:
    python -m src.generate_demo_dataset --samples-per-class 120
"""

import argparse
import csv

import numpy as np

from src import config
from src.feature_extraction import HandLandmarkExtractor
from src.utils import get_logger, safe_run

logger = get_logger(__name__)


def _synthetic_class_center(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    center = rng.uniform(-1, 1, size=config.FEATURE_VECTOR_LENGTH)
    norm = np.linalg.norm(center)
    return center / norm if norm > 0 else center


@safe_run(logger)
def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic landmark dataset for pipeline smoke-testing."
    )
    parser.add_argument("--classes", nargs="+", default=config.DEFAULT_CLASSES)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.08)
    args = parser.parse_args()

    config.ensure_directories()
    rows = []
    feature_names = HandLandmarkExtractor.feature_names()

    for class_idx, label in enumerate(args.classes):
        center = _synthetic_class_center(seed=class_idx * 17 + 3)
        rng = np.random.RandomState(class_idx * 101 + 7)
        samples = center + rng.normal(0, args.noise, size=(args.samples_per_class, len(center)))
        for sample in samples:
            rows.append(list(sample) + [label])

    with open(config.LANDMARKS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(feature_names + ["label"])
        writer.writerows(rows)

    logger.info("Wrote synthetic dataset with %d samples across %d classes to %s",
                len(rows), len(args.classes), config.LANDMARKS_CSV)
    print(f"Synthetic demo dataset written to {config.LANDMARKS_CSV} "
          f"({len(rows)} samples, classes: {', '.join(args.classes)})")
    print("NOTE: this is synthetic data for testing the pipeline only. "
          "Run collect_data.py with a real camera/images before real use.")


if __name__ == "__main__":
    main()
