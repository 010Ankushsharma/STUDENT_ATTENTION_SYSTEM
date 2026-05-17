
"""
eye_tracking/ear_calculator.py
Eye Aspect Ratio (EAR) — the core metric for blink & drowsiness detection.

Formula (Soukupová & Čech, 2016):
                |p2 - p6| + |p3 - p5|
    EAR  =  ─────────────────────────────
                  2 × |p1 - p4|

Landmark layout:
        p2    p3
    p1 ──────── p4
        p6    p5

Typical values:
    ┌──────────────┬───────────┐
    │ State        │ EAR Range │
    ├──────────────┼───────────┤
    │ Wide open    │ 0.30-0.38 │
    │ Normal open  │ 0.25-0.30 │
    │ Half closed  │ 0.18-0.25 │
    │ Fully closed │ 0.05-0.15 │
    └──────────────┴───────────┘
"""
import numpy as np
from typing import Tuple, List, Optional
from .config import cfg


class EARCalculator:
    """
    Computes Eye Aspect Ratio from MediaPipe Face Mesh landmarks.

    Usage:
        calc = EARCalculator()
        left_ear, right_ear, avg_ear = calc.compute(landmarks, w, h)
    """

    def __init__(self):
        self._left_idx = cfg.left_eye_idx  # [362,385,387,263,373,380]
        self._right_idx = cfg.right_eye_idx  # [33,160,158,133,153,144]

        # History for smoothing (reduces noise)
        self._history: List[float] = []
        self._history_size = 5

    # ──────────── Public API ────────────

    def compute(self, landmarks, frame_w: int, frame_h: int
                ) -> Tuple[float, float, float]:
        """
        Compute EAR for both eyes.

        Args:
            landmarks: MediaPipe face_landmarks.landmark list
            frame_w:   Frame width in pixels
            frame_h:   Frame height in pixels

        Returns:
            (left_ear, right_ear, average_ear) — all floats in [0, ~0.5]
        """
        left_ear = self._single_ear(landmarks, self._left_idx, frame_w, frame_h)
        right_ear = self._single_ear(landmarks, self._right_idx, frame_w, frame_h)
        avg_ear = (left_ear + right_ear) / 2.0

        # Smooth with moving average
        avg_ear = self._smooth(avg_ear)

        return round(left_ear, 4), round(right_ear, 4), round(avg_ear, 4)

    def compute_raw(self, landmarks, frame_w: int, frame_h: int) -> float:
        """Compute average EAR without smoothing (for per-frame analysis)."""
        left = self._single_ear(landmarks, self._left_idx, frame_w, frame_h)
        right = self._single_ear(landmarks, self._right_idx, frame_w, frame_h)
        return (left + right) / 2.0

    def get_eye_points(self, landmarks, frame_w: int, frame_h: int
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get pixel coordinates for both eye landmark sets.

        Returns:
            (left_points[6,2], right_points[6,2]) as numpy arrays
        """
        left = self._extract_points(landmarks, self._left_idx, frame_w, frame_h)
        right = self._extract_points(landmarks, self._right_idx, frame_w, frame_h)
        return left, right

    def reset(self):
        """Clear smoothing history."""
        self._history.clear()

    # ──────────── Internal ────────────

    @staticmethod
    def _extract_points(landmarks, indices: List[int],
                        w: int, h: int) -> np.ndarray:
        """Convert landmark indices to pixel coordinate array."""
        return np.array([
            (landmarks[i].x * w, landmarks[i].y * h)
            for i in indices
        ], dtype=np.float64)

    @staticmethod
    def _single_ear(landmarks, indices: List[int],
                    w: int, h: int) -> float:
        """
        Compute EAR for one eye.

        Landmark order must be: [p1, p2, p3, p4, p5, p6]
            p1 = outer corner     p4 = inner corner
            p2 = upper-outer      p5 = lower-inner
            p3 = upper-inner      p6 = lower-outer
        """
        pts = np.array([
            (landmarks[i].x * w, landmarks[i].y * h)
            for i in indices
        ], dtype=np.float64)

        # Vertical distances (two pairs across the eyelid)
        v1 = np.linalg.norm(pts[1] - pts[5])  # |p2 - p6|
        v2 = np.linalg.norm(pts[2] - pts[4])  # |p3 - p5|

        # Horizontal distance (eye width)
        hz = np.linalg.norm(pts[0] - pts[3])  # |p1 - p4|

        # EAR formula
        ear = (v1 + v2) / (2.0 * hz + 1e-7)
        return ear

    def _smooth(self, value: float) -> float:
        """Moving average smoothing to reduce frame-to-frame jitter."""
        self._history.append(value)
        if len(self._history) > self._history_size:
            self._history.pop(0)
        return sum(self._history) / len(self._history)

