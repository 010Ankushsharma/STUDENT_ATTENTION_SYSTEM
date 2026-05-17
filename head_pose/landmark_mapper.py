
"""
head_pose/landmark_mapper.py
Extracts and maps MediaPipe Face Mesh landmarks to 2D image points
and 3D model points for solvePnP.

MediaPipe Face Mesh Landmark Map (key points):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│         105●────────────────────────●334                        │
│        (L eyebrow inner)     (R eyebrow inner)                 │
│                    6●                                           │
│                (nose bridge)                                    │
│    263●─────468●          ●473─────●33                          │
│   (L eye)  (L iris)    (R iris)  (R eye)                       │
│                                                                 │
│    132●           4●            ●361                            │
│   (L cheek)   (nose bottom)   (R cheek)                        │
│                    1●                                           │
│                 (NOSE TIP)                                      │
│                                                                 │
│         287●───────────────●57                                  │
│        (L mouth)        (R mouth)                              │
│                                                                 │
│                   152●                                          │
│                  (CHIN)                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from .config import cfg


@dataclass
class MappedLandmarks:
    """Extracted 2D + 3D landmark pairs ready for solvePnP."""
    image_points_2d: np.ndarray  # Shape (N, 2), float64
    model_points_3d: np.ndarray  # Shape (N, 3), float64
    nose_tip_2d: Tuple[int, int]  # Pixel coords of nose tip
    all_points_2d: Optional[np.ndarray] = None  # For visualisation
    landmark_names: Optional[List[str]] = None


class LandmarkMapper:
    """
    Extracts specific landmarks from MediaPipe Face Mesh results
    and prepares them for pose estimation.

    Two modes:
      - Basic 6-point model (fast, decent accuracy)
      - Extended 14-point model (slower, better accuracy)

    Usage:
        mapper = LandmarkMapper()
        mapped = mapper.extract(face_landmarks.landmark, frame_w, frame_h)
        # mapped.image_points_2d → for solvePnP
        # mapped.model_points_3d → corresponding 3D model
    """

    def __init__(self, use_extended: bool = None):
        self._use_extended = use_extended if use_extended is not None else cfg.use_extended_model

        if self._use_extended:
            self._indices = cfg.extended_landmark_indices
            self._model_3d = cfg.extended_model_points_3d.copy()
            self._names = [
                "Nose tip", "Chin", "L eye outer", "R eye outer",
                "L mouth", "R mouth", "L iris", "R iris",
                "Nose bridge bot", "Nose bridge top",
                "L eyebrow", "R eyebrow", "L cheek", "R cheek",
            ]
        else:
            self._indices = cfg.pose_landmark_indices
            self._model_3d = cfg.model_points_3d.copy()
            self._names = [
                "Nose tip", "Chin", "L eye outer",
                "R eye outer", "L mouth", "R mouth",
            ]

    def extract(self, landmarks, frame_w: int, frame_h: int) -> MappedLandmarks:
        """
        Extract 2D image points from face landmarks.

        Args:
            landmarks:  MediaPipe face_landmarks.landmark list
            frame_w:    Frame width in pixels
            frame_h:    Frame height in pixels

        Returns:
            MappedLandmarks with matched 2D/3D point pairs.
        """
        image_points = np.array([
            (landmarks[idx].x * frame_w, landmarks[idx].y * frame_h)
            for idx in self._indices
        ], dtype=np.float64)

        nose_idx = self._indices[0]  # Always first = nose tip
        nose_tip = (
            int(landmarks[nose_idx].x * frame_w),
            int(landmarks[nose_idx].y * frame_h),
        )

        return MappedLandmarks(
            image_points_2d=image_points,
            model_points_3d=self._model_3d,
            nose_tip_2d=nose_tip,
            all_points_2d=image_points,
            landmark_names=self._names,
        )

    def extract_all_contour(self, landmarks, frame_w: int, frame_h: int) -> np.ndarray:
        """
        Extract face oval contour landmarks for wireframe drawing.
        MediaPipe face oval: indices around the jaw/forehead.
        """
        oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109, 10,
        ]
        return np.array([
            (int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h))
            for i in oval_indices
        ], dtype=np.int32)

    @property
    def point_count(self) -> int:
        return len(self._indices)

    @property
    def mode(self) -> str:
        return "extended_14pt" if self._use_extended else "basic_6pt"

