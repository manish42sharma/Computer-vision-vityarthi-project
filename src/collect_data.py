"""
collect_data.py
----------------
Functional Module 1: Data Collection & Processing.

Builds the labelled landmark dataset (data/processed/landmarks_dataset.csv)
that model training consumes. Two input modes are supported, and BOTH are
fully command-line driven:

  1. --source images   : walk a folder of images arranged as
                          data/raw/<CLASS_LABEL>/*.jpg and extract one
                          feature vector per image. This is the
                          recommended, fully headless mode for CI /
                          servers / evaluation machines with no camera.

  2. --source webcam    : capture live frames from a webcam for a
                          fixed duration per class. This mode opens a
                          preview window ONLY if --display is passed;
                          by default it runs headless and reports
                          progress on stdout, so it still satisfies the
                          "command-line executable" requirement on
                          machines that do have a webcam.

Output: a single CSV with columns [x0..z20, label], one row per
detected hand pose - the clear input/output structure required by the
project brief.
"""

import argparse
import csv
import os
import time

import cv2

from src import config
from src.feature_extraction import HandLandmarkExtractor
from src.utils import get_logger, safe_run, timeit, validate_path_exists

logger = get_logger(__name__)


def _draw_landmarks_if_supported(frame, landmarks):
    """Best-effort overlay for --display mode; harmless no-op if the
    installed mediapipe version doesn't expose the classic drawing_utils
    helper (see feature_extraction.py for the backend-detection logic)."""
    try:
        import mediapipe as mp
        mp.solutions.drawing_utils.draw_landmarks(
            frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS
        )
    except Exception:
        logger.debug("Skipping landmark overlay: drawing_utils not available on this backend.")


def _write_rows(rows, append: bool):
    config.ensure_directories()
    mode = "a" if append and os.path.exists(config.LANDMARKS_CSV) else "w"
    write_header = mode == "w"

    with open(config.LANDMARKS_CSV, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(HandLandmarkExtractor.feature_names() + ["label"])
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), config.LANDMARKS_CSV)


@timeit(logger)
def collect_from_images(raw_dir: str, append: bool = False) -> int:
    """
    Expects raw_dir/<label>/*.{jpg,png,jpeg}. Returns the number of
    samples successfully extracted.
    """
    validate_path_exists(raw_dir, "Raw images directory")
    class_dirs = sorted(
        d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))
    )
    if not class_dirs:
        raise ValueError(
            f"No class sub-folders found under {raw_dir}. "
            "Expected structure: data/raw/<LABEL>/*.jpg"
        )

    rows = []
    with HandLandmarkExtractor(static_image_mode=True) as extractor:
        for label in class_dirs:
            label_dir = os.path.join(raw_dir, label)
            image_files = [
                f for f in sorted(os.listdir(label_dir))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            logger.info("Processing class '%s' (%d images)", label, len(image_files))

            hits = 0
            for fname in image_files:
                path = os.path.join(label_dir, fname)
                frame = cv2.imread(path)
                if frame is None:
                    logger.warning("Could not read image %s; skipping.", path)
                    continue
                vector, _ = extractor.extract(frame)
                if vector is None:
                    logger.warning("No hand detected in %s; skipping.", path)
                    continue
                rows.append(list(vector) + [label])
                hits += 1
            logger.info("Class '%s': %d/%d images yielded a hand pose",
                         label, hits, len(image_files))

    if not rows:
        raise ValueError("No usable samples were extracted from any image.")

    _write_rows(rows, append)
    return len(rows)


@timeit(logger)
def collect_from_webcam(label: str, seconds: int, camera_index: int, display: bool,
                         append: bool = True) -> int:
    """
    Capture frames from a webcam for `seconds` seconds, label every
    detected hand pose as `label`, and append to the dataset CSV.
    Runs headless unless --display is explicitly requested.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "If no camera is available, use --source images instead."
        )

    rows = []
    start = time.time()
    frame_count = 0

    logger.info("Recording class '%s' for %d seconds (camera %d)...",
                label, seconds, camera_index)

    with HandLandmarkExtractor(static_image_mode=False) as extractor:
        try:
            while time.time() - start < seconds:
                ok, frame = cap.read()
                if not ok:
                    logger.warning("Frame grab failed; stopping capture early.")
                    break

                frame_count += 1
                if frame_count % config.FRAME_SKIP != 0:
                    continue

                vector, landmarks = extractor.extract(frame)
                if vector is not None:
                    rows.append(list(vector) + [label])

                if display:
                    if landmarks is not None:
                        _draw_landmarks_if_supported(frame, landmarks)
                    cv2.imshow("Data Collection (press q to stop early)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

    if not rows:
        raise ValueError("No hand poses were captured. Try again with better lighting/framing.")

    _write_rows(rows, append)
    logger.info("Captured %d samples for class '%s'.", len(rows), label)
    return len(rows)


@safe_run(logger)
def main():
    parser = argparse.ArgumentParser(
        description="Collect labelled hand-landmark samples for the "
                     "Sign Language Recognition dataset."
    )
    parser.add_argument("--source", choices=["images", "webcam"], required=True,
                         help="Where to read samples from.")
    parser.add_argument("--raw-dir", default=config.DATA_RAW_DIR,
                         help="Root folder of data/raw/<label>/*.jpg (images mode).")
    parser.add_argument("--label", help="Class label to record (webcam mode).")
    parser.add_argument("--seconds", type=int, default=15,
                         help="Recording duration in seconds (webcam mode).")
    parser.add_argument("--camera-index", type=int, default=0,
                         help="OpenCV camera index (webcam mode).")
    parser.add_argument("--display", action="store_true",
                         help="Show a live preview window (requires a GUI/display).")
    parser.add_argument("--append", action="store_true",
                         help="Append to the existing dataset instead of overwriting it.")
    args = parser.parse_args()

    if args.source == "images":
        n = collect_from_images(args.raw_dir, append=args.append)
    else:
        if not args.label:
            parser.error("--label is required when --source webcam is used.")
        n = collect_from_webcam(args.label, args.seconds, args.camera_index,
                                 args.display, append=True)

    print(f"Done. {n} labelled samples written to {config.LANDMARKS_CSV}")


if __name__ == "__main__":
    main()
