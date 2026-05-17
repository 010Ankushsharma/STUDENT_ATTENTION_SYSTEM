
"""
face_detection/config.py
Centralised configuration — every magic number lives here.
"""
from dataclasses import dataclass


@dataclass
class DetectionConfig:
    """MediaPipe + OpenCV detection parameters."""

    # --- MediaPipe Face Detection ---
    model_selection: int = 1  # 0 = short-range (2m), 1 = full-range (5m)
    min_detection_confidence: float = 0.5

    # --- MediaPipe Face Mesh (used for richer landmarks) ---
    use_face_mesh: bool = True
    max_num_faces: int = 15
    mesh_refine_landmarks: bool = True
    mesh_min_detection: float = 0.5
    mesh_min_tracking: float = 0.5

    # --- Frame Processing ---
    process_width: int = 640  # Downscale to this width for speed
    skip_frames: int = 0  # 0 = process every frame, 1 = every 2nd, etc.
    jpeg_quality: int = 75

    # --- Tracker ---
    max_disappeared: int = 30  # Frames before dropping a lost face
    max_distance: int = 80  # Pixel distance threshold for ID matching

    # --- Drawing ---
    bbox_thickness: int = 2
    font_scale: float = 0.55
    show_confidence: bool = True
    show_landmarks: bool = False  # Draw all 468 mesh points
    show_fps: bool = True

    # --- Lighting Robustness ---
    apply_clahe: bool = True  # Contrast-limited adaptive histogram eq.
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    # --- Performance ---
    use_gpu: bool = False  # MediaPipe GPU (requires special build)


# Singleton config instance — import this everywhere
cfg = DetectionConfig()
