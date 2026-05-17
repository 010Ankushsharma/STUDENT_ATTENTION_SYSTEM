
"""
head_pose/config.py
All tunable parameters for head pose estimation.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class HeadPoseConfig:
    """Centralised configuration for the head pose module."""

    # ─────────── MediaPipe ───────────
    max_faces: int = 10
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # ─────────── 3D Face Model Points ───────────
    #
    # Generic anthropometric model (mm). These correspond to
    # specific MediaPipe landmark indices defined below.
    #
    # Reference: OpenCV Head Pose Estimation literature
    #
    #   Point           3D Coordinate       Landmark
    #   ─────────────   ──────────────────   ────────
    #   Nose tip        ( 0,    0,     0  )  1
    #   Chin            ( 0, -330,  -65   )  152
    #   Left eye outer  (-225, 170, -135  )  263
    #   Right eye outer ( 225, 170, -135  )  33
    #   Left mouth      (-150,-150, -125  )  287
    #   Right mouth     ( 150,-150, -125  )  57
    #
    model_points_3d: np.ndarray = field(default_factory=lambda: np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye outer corner
        (225.0, 170.0, -135.0),  # Right eye outer corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ], dtype=np.float64))

    # Matching MediaPipe landmark indices (same order as model_points_3d)
    pose_landmark_indices: List[int] = field(default_factory=lambda: [
        1,  # Nose tip
        152,  # Chin
        263,  # Left eye outer corner
        33,  # Right eye outer corner
        287,  # Left mouth corner
        57,  # Right mouth corner
    ])

    # ─────────── Extended 14-Point Model (Higher Accuracy) ───────────
    #
    # More points = more constraints for solvePnP = better accuracy
    # but slightly slower. Toggle with `use_extended_model`.
    #
    use_extended_model: bool = True

    extended_model_points_3d: np.ndarray = field(default_factory=lambda: np.array([
        (0.0, 0.0, 0.0),  # 1   Nose tip
        (0.0, -330.0, -65.0),  # 152 Chin
        (-225.0, 170.0, -135.0),  # 263 Left eye outer
        (225.0, 170.0, -135.0),  # 33  Right eye outer
        (-150.0, -150.0, -125.0),  # 287 Left mouth corner
        (150.0, -150.0, -125.0),  # 57  Right mouth corner
        (-130.0, 170.0, -135.0),  # 468 Left iris (approx)
        (130.0, 170.0, -135.0),  # 473 Right iris (approx)
        (0.0, -90.0, 10.0),  # 4   Nose bridge bottom
        (0.0, 65.0, -10.0),  # 6   Nose bridge top
        (-100.0, 65.0, -100.0),  # 105 Left eyebrow inner
        (100.0, 65.0, -100.0),  # 334 Right eyebrow inner
        (-215.0, -100.0, -140.0),  # 132 Left cheek
        (215.0, -100.0, -140.0),  # 361 Right cheek
    ], dtype=np.float64))

    extended_landmark_indices: List[int] = field(default_factory=lambda: [
        1, 152, 263, 33, 287, 57,
        468, 473, 4, 6, 105, 334, 132, 361,
    ])

    # ─────────── Direction Thresholds (degrees) ───────────
    #
    #           LEFT          SLIGHT-LEFT     FORWARD    SLIGHT-RIGHT      RIGHT
    #    ◄──────|────────────────|──────────────|──────────────|────────────|──────►
    #         -30             -10              0             +10           +30       YAW
    #
    #            DOWN       SLIGHT-DOWN     FORWARD    SLIGHT-UP          UP
    #    ◄──────|────────────────|──────────────|──────────────|────────────|──────►
    #         -25              -8              0              +8           +25       PITCH
    #
    yaw_forward_range: Tuple[float, float] = (-10.0, 10.0)
    yaw_slight_range: Tuple[float, float] = (-30.0, 30.0)  # Beyond forward, within this
    pitch_forward_range: Tuple[float, float] = (-8.0, 8.0)
    pitch_slight_range: Tuple[float, float] = (-25.0, 25.0)
    roll_alert_threshold: float = 25.0  # Head tilt > this = tilted

    # ─────────── Attention / Teacher Position ───────────
    #
    # Teacher/screen assumed at yaw=0, pitch=0 (directly ahead).
    # Customise if camera is off-centre.
    #
    teacher_yaw: float = 0.0
    teacher_pitch: float = 0.0
    attention_yaw_tolerance: float = 20.0  # Degrees either side of teacher
    attention_pitch_tolerance: float = 15.0

    # ─────────── Smoothing ───────────
    smoothing_window: int = 5  # Moving average window (frames)
    use_kalman: bool = True  # Kalman filter for smoother tracking

    # ─────────── Visualization ───────────
    draw_3d_axes: bool = True  # RGB XYZ axes on nose
    draw_direction_label: bool = True
    draw_angle_gauges: bool = True
    draw_face_wireframe: bool = False  # Mesh wireframe overlay
    draw_landmark_points: bool = False  # Show raw 6/14 points
    axis_length: float = 80.0  # 3D axis line length (pixels)

    # ─────────── Performance ───────────
    process_width: int = 640
    solvepnp_method: int = 0  # 0=ITERATIVE, 1=EPNP, 2=SQPNP

    # Map method int to OpenCV flag (populated at runtime)
    @property
    def cv2_solvepnp_flag(self):
        import cv2
        flags = {
            0: cv2.SOLVEPNP_ITERATIVE,
            1: cv2.SOLVEPNP_EPNP,
            2: cv2.SOLVEPNP_SQPNP,
        }
        return flags.get(self.solvepnp_method, cv2.SOLVEPNP_ITERATIVE)


cfg = HeadPoseConfig()
