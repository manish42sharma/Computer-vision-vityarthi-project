"""
feature_extraction.py
----------------------
Core computer-vision module: converts a raw BGR image (a frame from a
webcam, a video file, or a static photo) into a fixed-length numeric
feature vector describing the hand's pose.

Design rationale (see docs/DESIGN.md for the full rationale):
    Rather than feeding raw pixels into a CNN (heavy, needs thousands of
    images per class, slow to train on a laptop), this project uses
    MediaPipe Hands to locate 21 3-D hand landmarks per frame and treats
    their normalised (x, y, z) coordinates as the feature vector. This is
    the standard lightweight approach for static-gesture / fingerspelling
    recognition and keeps the whole pipeline runnable on a CPU-only
    machine, from the command line, with no GPU or GUI required.
"""

import os
from typing import List, Optional, Tuple

import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mediapipe is required for feature extraction. "
        "Install project dependencies with: pip install -r requirements.txt"
    ) from exc

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

# MediaPipe ships two Python APIs for hand tracking depending on package
# version:
#   1. The classic "solutions" API (mp.solutions.hands) - used by the
#      vast majority of tutorials/course code and the one this project
#      targets by default (see requirements.txt for the pinned version).
#   2. The newer "Tasks" API (mediapipe.tasks.python.vision.HandLandmarker)
#      which some recent mediapipe releases ship exclusively. It needs an
#      explicit .task model file, which this module will look for at
#      MEDIAPIPE_HAND_MODEL_PATH (env var) or ./models/hand_landmarker.task.
#
# This module auto-detects whichever is available so the project keeps
# working across mediapipe releases without code changes.
_HAS_SOLUTIONS_API = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")


class HandLandmarkExtractor:
    """
    Thin, resource-managed wrapper around MediaPipe hand tracking.

    Usage:
        with HandLandmarkExtractor() as extractor:
            vector, landmarks = extractor.extract(frame)
    """

    def __init__(
        self,
        max_num_hands: int = config.MAX_NUM_HANDS,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
        static_image_mode: bool = False,
    ):
        self._backend = "solutions" if _HAS_SOLUTIONS_API else "tasks"

        if self._backend == "solutions":
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            self._init_tasks_backend(max_num_hands, min_detection_confidence,
                                      min_tracking_confidence, static_image_mode)

        logger.debug("HandLandmarkExtractor using '%s' MediaPipe backend.", self._backend)

    def _init_tasks_backend(self, max_num_hands, min_detection_confidence,
                             min_tracking_confidence, static_image_mode):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as base_options_lib

        model_path = os.environ.get(
            "MEDIAPIPE_HAND_MODEL_PATH",
            os.path.join(config.MODELS_DIR, "hand_landmarker.task"),
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "This mediapipe installation only exposes the newer Tasks API, "
                "which needs a hand_landmarker.task model file. Either:\n"
                "  (a) pip install a mediapipe version with the classic "
                "'solutions' API (see requirements.txt), or\n"
                "  (b) download the model from "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task and place it at "
                f"{model_path} (or set MEDIAPIPE_HAND_MODEL_PATH)."
            )

        base_options = base_options_lib.BaseOptions(model_asset_path=model_path)
        running_mode = (vision.RunningMode.IMAGE if static_image_mode
                         else vision.RunningMode.VIDEO)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._tasks_vision = vision
        self._hands = vision.HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._hands.close()

    # ------------------------------------------------------------------
    def extract(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[object]]:
        """
        Run hand detection on a single BGR frame.

        Returns:
            (feature_vector, landmarks) if a hand was found, otherwise
            (None, None). `landmarks` is a backend-specific object;
            callers needing to draw an overlay should use
            HandLandmarkExtractor.to_drawable(landmarks) if portability
            across backends is needed.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("Received an empty frame; skipping.")
            return None, None

        if self._backend == "solutions":
            return self._extract_solutions(frame_bgr)
        return self._extract_tasks(frame_bgr)

    def _extract_solutions(self, frame_bgr: np.ndarray):
        rgb_frame = frame_bgr[:, :, ::-1]  # BGR -> RGB, no extra copy of channels
        results = self._hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None, None

        hand_landmarks = results.multi_hand_landmarks[0]
        vector = self._landmarks_to_vector(hand_landmarks)
        return vector, hand_landmarks

    def _extract_tasks(self, frame_bgr: np.ndarray):
        rgb_frame = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))

        if self._hands._running_mode == self._tasks_vision.RunningMode.VIDEO:
            self._frame_timestamp_ms += 33  # assume ~30fps if not otherwise known
            result = self._hands.detect_for_video(mp_image, self._frame_timestamp_ms)
        else:
            result = self._hands.detect(mp_image)

        if not result.hand_landmarks:
            return None, None

        hand_landmarks = result.hand_landmarks[0]  # list of NormalizedLandmark
        vector = self._landmarks_to_vector_from_list(hand_landmarks)
        return vector, hand_landmarks

    # ------------------------------------------------------------------
    @staticmethod
    def _landmarks_to_vector(hand_landmarks) -> np.ndarray:
        """
        Flatten MediaPipe's 21 landmarks into a 63-length vector, then
        normalise translation and scale relative to the wrist and the
        hand's bounding size. Normalisation is what makes the model
        robust to where the hand sits in the frame and how close it is
        to the camera.
        """
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )

        wrist = coords[0].copy()
        coords -= wrist  # translation invariance

        max_extent = np.max(np.linalg.norm(coords, axis=1))
        if max_extent > 1e-6:
            coords /= max_extent  # scale invariance

        return coords.flatten()

    @staticmethod
    def _landmarks_to_vector_from_list(landmark_list) -> np.ndarray:
        """Same normalisation as _landmarks_to_vector, for the Tasks-API
        return type (a plain list of landmark objects rather than a
        message with a `.landmark` attribute)."""
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmark_list],
            dtype=np.float32,
        )
        wrist = coords[0].copy()
        coords -= wrist
        max_extent = np.max(np.linalg.norm(coords, axis=1))
        if max_extent > 1e-6:
            coords /= max_extent
        return coords.flatten()

    @staticmethod
    def feature_names() -> List[str]:
        names = []
        for i in range(config.NUM_LANDMARKS):
            names.extend([f"x{i}", f"y{i}", f"z{i}"])
        return names
