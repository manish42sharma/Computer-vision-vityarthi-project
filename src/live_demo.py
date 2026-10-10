"""
live_demo.py
------------
OPTIONAL bonus module. Everything required by the project brief is
already satisfied headlessly by predict.py (image/folder/video, CLI
only, no window needed). This module simply wraps SignPredictor in a
live webcam loop with an on-screen overlay, for anyone demoing the
project interactively on a machine that has both a camera and a
display. It is not part of the required command-line evaluation path.

Usage:
    python -m src.live_demo --camera-index 0
"""

import argparse

import cv2

from src.predict import SignPredictor
from src.utils import get_logger, safe_run

logger = get_logger(__name__)


def _draw_landmarks_if_supported(frame, landmarks):
    try:
        import mediapipe as mp
        mp.solutions.drawing_utils.draw_landmarks(
            frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS
        )
    except Exception:
        pass


@safe_run(logger)
def main():
    parser = argparse.ArgumentParser(description="Optional live webcam sign-recognition demo.")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}.")

    logger.info("Starting live demo. Press 'q' to quit.")
    with SignPredictor() as predictor:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                label, confidence, landmarks = predictor.predict_frame(frame)
                if landmarks is not None:
                    _draw_landmarks_if_supported(frame, landmarks)
                text = f"{label} ({confidence:.2f})" if label else "No hand detected"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)
                cv2.imshow("Sign Language Recognition - Live Demo (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
