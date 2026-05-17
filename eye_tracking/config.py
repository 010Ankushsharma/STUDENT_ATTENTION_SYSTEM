
"""
eye_tracking/config.py
All tunable parameters for the eye tracking module.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class EyeTrackingConfig:
    """Centralised configuration."""

    # ──────────── MediaPipe ────────────
    max_faces: int = 5
    refine_landmarks: bool = True  # Enables iris landmarks (478 total)
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # ──────────── EAR Thresholds ────────────
    #
    #   EAR (Eye Aspect Ratio) ranges:
    #     Open eye   : 0.25 – 0.35
    #     Closing    : 0.18 – 0.25
    #     Closed     : 0.05 – 0.18
    #
    ear_blink_threshold: float = 0.21  # Below this = eyes closed
    ear_partial_threshold: float = 0.26  # Below this = partially closed

    # ──────────── Blink Detection ────────────
    blink_consec_frames: int = 2  # Min consecutive closed frames = 1 blink
    max_blink_duration_frames: int = 8  # Beyond this = prolonged closure, not blink

    # ──────────── Drowsiness ────────────
    drowsy_ear_threshold: float = 0.20  # Sustained EAR below this = drowsy
    drowsy_consec_frames: int = 15  # Frames of low EAR to trigger drowsy
    drowsy_blink_rate_high: float = 25.0  # Blinks/min above this = fatigued
    drowsy_blink_rate_low: float = 5.0  # Blinks/min below this = staring/zoned out
    perclos_threshold: float = 0.15  # PERCLOS > 15% = drowsy
    perclos_window_frames: int = 300  # 10 seconds at 30 FPS

    # ──────────── Gaze ────────────
    gaze_horizontal_threshold: float = 0.60  # Iris ratio beyond this = looking away
    gaze_vertical_threshold: float = 0.58

    # ──────────── Visualisation ────────────
    draw_eye_contours: bool = True
    draw_iris: bool = True
    draw_ear_bar: bool = True
    draw_metrics_panel: bool = True
    show_gaze_arrow: bool = True

    # ──────────── Performance ────────────
    process_width: int = 640
    skip_frames: int = 0

    # ──────────── MediaPipe Landmark Indices ────────────
    # Left eye contour (upper + lower lids)
    left_eye_idx: List[int] = field(default_factory=lambda: [
        362, 385, 387, 263, 373, 380,  # EAR 6-point
    ])
    left_eye_contour_idx: List[int] = field(default_factory=lambda: [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
        388, 387, 386, 385, 384, 398,
    ])
    # Right eye contour
    right_eye_idx: List[int] = field(default_factory=lambda: [
        33, 160, 158, 133, 153, 144,  # EAR 6-point
    ])
    right_eye_contour_idx: List[int] = field(default_factory=lambda: [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
        157, 158, 159, 160, 161, 246,
    ])
    # Iris landmarks (available when refine_landmarks=True)
    left_iris_idx: List[int] = field(default_factory=lambda: [
        468, 469, 470, 471, 472,  # Center + 4 cardinal
    ])
    right_iris_idx: List[int] = field(default_factory=lambda: [
        473, 474, 475, 476, 477,  # Center + 4 cardinal
    ])


cfg = EyeTrackingConfig()
