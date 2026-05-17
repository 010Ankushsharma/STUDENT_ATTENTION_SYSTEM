
"""
head_pose/pose_calculator.py
Core head pose computation using OpenCV solvePnP.

Pipeline:
    2D image points + 3D model points + camera matrix
        → solvePnP → rotation vector (rvec)
        → Rodrigues → rotation matrix
        → RQDecomp3x3 → Euler angles (pitch, yaw, roll)

    Also computes:
        - 3D axis endpoints for visualisation
        - Nose direction vector for gaze line
        - Reprojection error for confidence

solvePnP Methods:
    ┌───────────────┬───────────────────────────────────┐
    │ Method        │ Notes                              │
    ├───────────────┼───────────────────────────────────┤
    │ ITERATIVE     │ Default. Good accuracy, moderate.  │
    │ EPNP          │ Faster, slightly less accurate.    │
    │ SQPNP         │ Newest. Good for noisy data.       │
    └───────────────┴───────────────────────────────────┘
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
from .config import cfg
from .landmark_mapper import MappedLandmarks


@dataclass
class PoseResult:
    """Complete head pose estimation result."""
    yaw: float  # Left(-) / Right(+) in degrees
    pitch: float  # Down(-) / Up(+) in degrees
    roll: float  # Tilt left(-) / right(+) in degrees
    rvec: np.ndarray  # Rotation vector (3,1)
    tvec: np.ndarray  # Translation vector (3,1)
    rmat: np.ndarray  # Rotation matrix (3,3)
    nose_tip_2d: Tuple[int, int]
    nose_end_2d: Tuple[int, int]  # Projected nose direction point
    axis_points_2d: Optional[np.ndarray] = None  # [origin, X, Y, Z] in 2D
    reprojection_error: float = 0.0
    confidence: float = 1.0  # 1.0 - normalised reprojection error


class PoseCalculator:
    """
    Computes 3D head pose from 2D-3D point correspondences.

    Features:
      - solvePnP with configurable method
      - Euler angle extraction (yaw, pitch, roll)
      - 3D axis projection for visualisation
      - Moving average + optional Kalman smoothing
      - Reprojection error for confidence

    Usage:
        calc = PoseCalculator(frame_w=640, frame_h=480)
        result = calc.compute(mapped_landmarks)
        print(f"Yaw: {result.yaw:.1f}°, Pitch: {result.pitch:.1f}°")
    """

    def __init__(self, frame_w: int = 640, frame_h: int = 480):
        self._w = frame_w
        self._h = frame_h

        # Camera intrinsics (approximate pinhole model)
        focal_length = frame_w
        cx, cy = frame_w / 2.0, frame_h / 2.0
        self._camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1],
        ], dtype=np.float64)

        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Smoothing buffers
        self._yaw_buf = deque(maxlen=cfg.smoothing_window)
        self._pitch_buf = deque(maxlen=cfg.smoothing_window)
        self._roll_buf = deque(maxlen=cfg.smoothing_window)

        # Kalman filter (for yaw and pitch)
        if cfg.use_kalman:
            self._kf_yaw = self._init_kalman()
            self._kf_pitch = self._init_kalman()
            self._kf_roll = self._init_kalman()

        # Previous rvec/tvec for iterative refinement
        self._prev_rvec: Optional[np.ndarray] = None
        self._prev_tvec: Optional[np.ndarray] = None

    # ──────────── Public API ────────────

    def compute(self, mapped: MappedLandmarks) -> Optional[PoseResult]:
        """
        Compute head pose from mapped landmarks.

        Args:
            mapped: MappedLandmarks from LandmarkMapper.extract()

        Returns:
            PoseResult with angles, vectors, and axis points.
            None if solvePnP fails.
        """
        image_pts = mapped.image_points_2d
        model_pts = mapped.model_points_3d

        # ── solvePnP ──
        if self._prev_rvec is not None:
            success, rvec, tvec = cv2.solvePnP(
                model_pts, image_pts,
                self._camera_matrix, self._dist_coeffs,
                rvec=self._prev_rvec.copy(),
                tvec=self._prev_tvec.copy(),
                useExtrinsicGuess=True,
                flags=cfg.cv2_solvepnp_flag,
            )
        else:
            success, rvec, tvec = cv2.solvePnP(
                model_pts, image_pts,
                self._camera_matrix, self._dist_coeffs,
                flags=cfg.cv2_solvepnp_flag,
            )

        if not success:
            return None

        # Store for next frame's initial guess
        self._prev_rvec = rvec.copy()
        self._prev_tvec = tvec.copy()

        # ── Rotation matrix & Euler angles ──
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        raw_pitch = float(angles[0])
        raw_yaw = float(angles[1])
        raw_roll = float(angles[2])

        # ── Smooth ──
        yaw = self._smooth(raw_yaw, self._yaw_buf, self._kf_yaw if cfg.use_kalman else None)
        pitch = self._smooth(raw_pitch, self._pitch_buf, self._kf_pitch if cfg.use_kalman else None)
        roll = self._smooth(raw_roll, self._roll_buf, self._kf_roll if cfg.use_kalman else None)

        # ── Project 3D axes for visualisation ──
        axis_points_2d = self._project_axes(rvec, tvec)

        # ── Nose direction endpoint ──
        nose_end_3d = np.array([(0.0, 0.0, 500.0)], dtype=np.float64)
        nose_end_2d, _ = cv2.projectPoints(
            nose_end_3d, rvec, tvec,
            self._camera_matrix, self._dist_coeffs,
        )
        nose_end = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))

        # ── Reprojection error ──
        reproj_err = self._reprojection_error(model_pts, image_pts, rvec, tvec)
        confidence = max(0.0, 1.0 - reproj_err / 20.0)  # Normalise

        return PoseResult(
            yaw=round(yaw, 1),
            pitch=round(pitch, 1),
            roll=round(roll, 1),
            rvec=rvec,
            tvec=tvec,
            rmat=rmat,
            nose_tip_2d=mapped.nose_tip_2d,
            nose_end_2d=nose_end,
            axis_points_2d=axis_points_2d,
            reprojection_error=round(reproj_err, 2),
            confidence=round(confidence, 2),
        )

    def update_frame_size(self, w: int, h: int):
        """Call if frame dimensions change."""
        self._w = w
        self._h = h
        focal = w
        cx, cy = w / 2.0, h / 2.0
        self._camera_matrix = np.array([
            [focal, 0, cx], [0, focal, cy], [0, 0, 1],
        ], dtype=np.float64)

    def reset(self):
        """Clear smoothing buffers and previous pose."""
        self._yaw_buf.clear()
        self._pitch_buf.clear()
        self._roll_buf.clear()
        self._prev_rvec = None
        self._prev_tvec = None
        if cfg.use_kalman:
            self._kf_yaw = self._init_kalman()
            self._kf_pitch = self._init_kalman()
            self._kf_roll = self._init_kalman()

    # ──────────── Internal ────────────

    def _project_axes(self, rvec, tvec) -> np.ndarray:
        """Project XYZ axes from nose tip for 3D orientation visualisation."""
        length = cfg.axis_length
        axes_3d = np.array([
            [0, 0, 0],  # Origin (nose tip)
            [length, 0, 0],  # X axis (red)
            [0, length, 0],  # Y axis (green)  [note: down in image]
            [0, 0, length],  # Z axis (blue - forward)
        ], dtype=np.float64)

        projected, _ = cv2.projectPoints(
            axes_3d, rvec, tvec,
            self._camera_matrix, self._dist_coeffs,
        )
        return projected.reshape(-1, 2).astype(int)

    def _reprojection_error(self, model_pts, image_pts, rvec, tvec) -> float:
        """Compute mean reprojection error (pixels)."""
        projected, _ = cv2.projectPoints(
            model_pts, rvec, tvec,
            self._camera_matrix, self._dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - image_pts, axis=1)
        return float(np.mean(errors))

    @staticmethod
    def _smooth(value: float, buf: deque, kalman=None) -> float:
        """Apply moving average + optional Kalman filtering."""
        buf.append(value)
        ma = sum(buf) / len(buf)

        if kalman is not None:
            kalman.correct(np.array([[np.float32(ma)]]))
            prediction = kalman.predict()
            return float(prediction[0][0])
        return ma

    @staticmethod
    def _init_kalman():
        """Initialise a 1D Kalman filter for angle smoothing."""
        kf = cv2.KalmanFilter(2, 1)  # state=[angle, velocity], measurement=[angle]
        kf.measurementMatrix = np.array([[1, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        kf.processNoiseCov = np.eye(2, dtype=np.float32) * 1e-3
        kf.measurementNoiseCov = np.array([[5e-2]], np.float32)
        kf.errorCovPost = np.eye(2, dtype=np.float32)
        return kf

