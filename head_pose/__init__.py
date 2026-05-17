"""
Head Pose Estimation Module
============================
Estimates 3D head orientation (yaw, pitch, roll) using MediaPipe
Face Mesh landmarks and OpenCV solvePnP.

Core Components:
    - LandmarkMapper      : 2D/3D landmark extraction & mapping
    - PoseCalculator       : solvePnP-based angle computation
    - DirectionClassifier  : Angle → direction label (forward/left/right/up/down)
    - AttentionJudge       : Is student facing the teacher/screen?
    - PoseVisualizer       : 3D axis, direction labels, gauge overlays
    - HeadPosePipeline     : End-to-end orchestrator

Usage:
    from head_pose import HeadPosePipeline
    pipeline = HeadPosePipeline()
    pipeline.run()                          # live webcam
    # OR
    results, frame = pipeline.process_frame(bgr_frame)
"""
from .landmark_mapper import LandmarkMapper
from .pose_calculator import PoseCalculator
from .direction_classifier import DirectionClassifier
from .attention_judge import AttentionJudge
from .visualizer import PoseVisualizer
from .pipeline import HeadPosePipeline

__all__ = [
    "LandmarkMapper",
    "PoseCalculator",
    "DirectionClassifier",
    "AttentionJudge",
    "PoseVisualizer",
    "HeadPosePipeline",
]
__version__ = "1.0.0"
