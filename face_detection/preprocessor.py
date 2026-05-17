
"""
face_detection/preprocessor.py
Frame pre-processing for lighting robustness and performance.
"""
import cv2
import numpy as np
from .config import cfg


class FramePreprocessor:
    """
    Prepares raw webcam frames before detection:
      1. Resize for speed
      2. CLAHE for lighting normalisation
      3. Optional denoising
    """

    def __init__(self):
        if cfg.apply_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=cfg.clahe_clip,
                tileGridSize=(cfg.clahe_grid, cfg.clahe_grid),
            )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply all pre-processing steps.

        Args:
            frame: Raw BGR frame from webcam.

        Returns:
            Processed BGR frame ready for detection.
        """
        # 1. Resize (maintain aspect ratio)
        frame = self._resize(frame)

        # 2. CLAHE on luminance channel for lighting robustness
        if cfg.apply_clahe:
            frame = self._apply_clahe(frame)

        return frame

    # ---- internal helpers ----

    @staticmethod
    def _resize(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        target_w = cfg.process_width
        if w <= target_w:
            return frame
        ratio = target_w / w
        new_h = int(h * ratio)
        return cv2.resize(frame, (target_w, new_h), interpolation=cv2.INTER_AREA)

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L-channel of LAB colour space."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

