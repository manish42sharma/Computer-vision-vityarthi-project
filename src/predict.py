"""
predict.py
----------
Functional Module 3: Sign Recognition (Inference).

The primary user-facing entry point. Fully usable from a terminal with
no GUI: give it an image, a folder of images, or a video file and it
prints the recognised sign(s) to stdout and can optionally write a CSV
of per-frame predictions. A --display flag additionally opens an
annotated preview window for machines that do have a screen, but this
is strictly optional - the command produces a correct, complete result
without it.

Examples
--------
    python -m src.predict --input samples/test_A.jpg
    python -m src.predict --input samples/ --mode folder
    python -m src.predict --input samples/demo.mp4 --mode video --output-csv out.csv
"""

import argparse
import csv
import os
import time
from collections import Counter

import cv2
import numpy as np

from src import config
from src.feature_extraction import HandLandmarkExtractor
from src.model import load_model
from src.utils import get_logger, safe_run, timeit, validate_path_exists

logger = get_logger(__name__)


class SignPredictor:
    """
    Wraps a trained model + label encoder + landmark extractor into one
    object with a simple `predict_frame` API, so predict.py, evaluate.py
    and any future integration (e.g. a REST endpoint) reuse identical
    logic instead of re-implementing inference.
    """

    def __init__(self):
        validate_path_exists(config.MODEL_FILE, "Trained model. Run train_model.py first.")
        self.model, self.encoder = load_model()
        self.extractor = HandLandmarkExtractor(static_image_mode=True)

    def close(self):
        self.extractor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def predict_frame(self, frame: np.ndarray):
        """
        Returns (label:str|None, confidence:float, landmarks) for a
        single BGR frame. label is None if no hand was detected, or if
        confidence falls below config.CONFIDENCE_THRESHOLD.
        """
        vector, landmarks = self.extractor.extract(frame)
        if vector is None:
            return None, 0.0, None

        probabilities = self._predict_proba(vector)
        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])
        label = self.encoder.inverse_transform([best_idx])[0]

        if confidence < config.CONFIDENCE_THRESHOLD:
            return "Unknown", confidence, landmarks
        return str(label), confidence, landmarks

    def _predict_proba(self, vector: np.ndarray) -> np.ndarray:
        vector = vector.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(vector)[0]
        # Fallback for models without predict_proba: one-hot confidence.
        pred = self.model.predict(vector)[0]
        probs = np.zeros(len(self.encoder.classes_))
        probs[pred] = 1.0
        return probs


@timeit(logger)
def predict_image(predictor: SignPredictor, image_path: str):
    validate_path_exists(image_path, "Input image")
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")
    label, confidence, _ = predictor.predict_frame(frame)
    return [{"source": image_path, "label": label, "confidence": round(confidence, 4)}]


@timeit(logger)
def predict_folder(predictor: SignPredictor, folder_path: str):
    validate_path_exists(folder_path, "Input folder")
    results = []
    files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not files:
        raise ValueError(f"No image files found in {folder_path}")

    for fname in files:
        path = os.path.join(folder_path, fname)
        frame = cv2.imread(path)
        if frame is None:
            logger.warning("Skipping unreadable file: %s", path)
            continue
        label, confidence, _ = predictor.predict_frame(frame)
        results.append({"source": path, "label": label, "confidence": round(confidence, 4)})
    return results


@timeit(logger)
def predict_video(predictor: SignPredictor, video_path: str, display: bool = False,
                   frame_skip: int = config.FRAME_SKIP):
    validate_path_exists(video_path, "Input video")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    results = []
    frame_idx = 0
    processed = 0

    try:
        while processed < config.MAX_VIDEO_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            label, confidence, landmarks = predictor.predict_frame(frame)
            results.append({
                "source": f"{video_path}#frame{frame_idx}",
                "label": label,
                "confidence": round(confidence, 4),
            })
            processed += 1

            if display:
                display_frame = frame.copy()
                text = f"{label or '...'} ({confidence:.2f})" if label else "No hand detected"
                cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)
                cv2.imshow("Sign Recognition (press q to quit)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

    return results


def summarise(results):
    labelled = [r["label"] for r in results if r["label"] and r["label"] != "Unknown"]
    if not labelled:
        return None
    most_common_label, count = Counter(labelled).most_common(1)[0]
    return {
        "majority_label": most_common_label,
        "majority_count": count,
        "total_frames_with_hand": len(labelled),
        "total_frames_processed": len(results),
    }


def write_csv(results, output_csv: str):
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "label", "confidence"])
        writer.writeheader()
        writer.writerows(results)
    logger.info("Wrote %d predictions to %s", len(results), output_csv)


@safe_run(logger)
def main():
    parser = argparse.ArgumentParser(
        description="Recognise a sign-language gesture from an image, "
                     "folder of images, or a video file - fully from the CLI."
    )
    parser.add_argument("--input", required=True, help="Path to image, folder, or video.")
    parser.add_argument("--mode", choices=["auto", "image", "folder", "video"], default="auto")
    parser.add_argument("--output-csv", default=None, help="Optional path to save results as CSV.")
    parser.add_argument("--display", action="store_true",
                         help="Show an annotated preview window (video mode only).")
    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        if os.path.isdir(args.input):
            mode = "folder"
        elif args.input.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            mode = "video"
        else:
            mode = "image"

    start = time.perf_counter()
    with SignPredictor() as predictor:
        if mode == "image":
            results = predict_image(predictor, args.input)
        elif mode == "folder":
            results = predict_folder(predictor, args.input)
        else:
            results = predict_video(predictor, args.input, display=args.display)
    elapsed = time.perf_counter() - start

    print("\n=== Recognition Results ===")
    for r in results:
        print(f"{r['source']:<45} -> {r['label'] or 'No hand detected':<10} "
              f"(confidence: {r['confidence']:.2f})")

    summary = summarise(results)
    if summary:
        print(f"\nMost likely sign: '{summary['majority_label']}' "
              f"({summary['majority_count']}/{summary['total_frames_with_hand']} "
              f"detected frames agree)")
    else:
        print("\nNo confident sign was recognised in the given input.")

    print(f"\nProcessed {len(results)} frame(s)/image(s) in {elapsed:.2f}s "
          f"({len(results)/elapsed if elapsed > 0 else 0:.1f} fps).")

    if args.output_csv:
        write_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
