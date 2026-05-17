
"""
eye_tracking/gaze_estimator.py
Iris-based gaze direction estimation using MediaPipe's refined landmarks.

How it works:
    MediaPipe provides 5 iris landmarks per eye (indices 468-477).
    We compute the iris center position relative to the eye bounding box:

    ratio = (iris_center - eye_inner) / (eye_outer - eye_inner)

    ratio ≈ 0.5 → looking straight
    ratio < 0.4 → looking left (for left eye)
    ratio > 0.6 → looking right

    Same logic vertically for up/down.
"""
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from .config import cfg


@dataclass
class GazeResult:
    """Gaze direction result."""
    horizontal: str  # "center", "left", "right"
    vertical: str  # "center", "up", "down"
    h_ratio: float  # 0.0 (leftmost) to 1.0 (rightmost)
    v_ratio: float  # 0.0 (top) to 1.0 (bottom)
    is_looking_away: bool  # True if gaze deviates significantly
    direction_label: str  # Combined: "center", "up-left", etc.


class GazeEstimator:
    """
    Estimates gaze direction from iris landmarks.

    Requires refine_landmarks=True in MediaPipe Face Mesh
    (provides landmarks 468-477 for iris positions).

    Usage:
        estimator = GazeEstimator()
        gaze = estimator.estimate(landmarks, frame_w, frame_h)
        print(gaze.direction_label, gaze.is_looking_away)
    """

    def estimate(self, landmarks, frame_w: int, frame_h: int) -> GazeResult:
        """
        Estimate gaze direction from iris position relative to eye box.

        Args:
            landmarks: face_landmarks.landmark list (478 points)
            frame_w, frame_h: Frame dimensions

        Returns:
            GazeResult with direction and ratios.
        """
        # Left eye: iris center = 468, corners = 362 (outer), 263 (inner)
        # Right eye: iris center = 473, corners = 33 (inner), 133 (outer)  [note: mirrored]
        # Upper/lower for vertical: left=[386(up),374(down)], right=[159(up),145(down)]

        # ── Left eye horizontal ratio ──
        l_iris = self._pt(landmarks, 468, frame_w, frame_h)
        l_outer = self._pt(landmarks, 362, frame_w, frame_h)
        l_inner = self._pt(landmarks, 263, frame_w, frame_h)
        l_upper = self._pt(landmarks, 386, frame_w, frame_h)
        l_lower = self._pt(landmarks, 374, frame_w, frame_h)

        l_h_ratio = self._ratio(l_iris[0], l_outer[0], l_inner[0])
        l_v_ratio = self._ratio(l_iris[1], l_upper[1], l_lower[1])

        # ── Right eye horizontal ratio ──
        r_iris = self._pt(landmarks, 473, frame_w, frame_h)
        r_inner = self._pt(landmarks, 33, frame_w, frame_h)
        r_outer = self._pt(landmarks, 133, frame_w, frame_h)
        r_upper = self._pt(landmarks, 159, frame_w, frame_h)
        r_lower = self._pt(landmarks, 145, frame_w, frame_h)

        r_h_ratio = self._ratio(r_iris[0], r_inner[0], r_outer[0])
        r_v_ratio = self._ratio(r_iris[1], r_upper[1], r_lower[1])

        # ── Average both eyes ──
        h_ratio = (l_h_ratio + r_h_ratio) / 2.0
        v_ratio = (l_v_ratio + r_v_ratio) / 2.0

        # ── Classify direction ──
        h_thresh = cfg.gaze_horizontal_threshold
        v_thresh = cfg.gaze_vertical_threshold

        if h_ratio < (1.0 - h_thresh):
            horizontal = "left"
        elif h_ratio > h_thresh:
            horizontal = "right"
        else:
            horizontal = "center"

        if v_ratio < (1.0 - v_thresh):
            vertical = "up"
        elif v_ratio > v_thresh:
            vertical = "down"
        else:
            vertical = "center"

        # Combined label
        if horizontal == "center" and vertical == "center":
            label = "center"
        elif horizontal == "center":
            label = vertical
        elif vertical == "center":
            label = horizontal
        else:
            label = f"{vertical}-{horizontal}"

        is_looking_away = horizontal != "center" or vertical != "center"

        return GazeResult(
            horizontal=horizontal,
            vertical=vertical,
            h_ratio=round(h_ratio, 3),
            v_ratio=round(v_ratio, 3),
            is_looking_away=is_looking_away,
            direction_label=label,
        )

    @staticmethod
    def _pt(landmarks, idx: int, w: int, h: int) -> Tuple[float, float]:
        """Extract pixel coordinates for a landmark."""
        return (landmarks[idx].x * w, landmarks[idx].y * h)

    @staticmethod
    def _ratio(value: float, start: float, end: float) -> float:
        """Compute normalised position ratio (0 to 1)."""
        denom = abs(end - start)
        if denom < 1e-6:
            return 0.5
        return (value - start) / denom

