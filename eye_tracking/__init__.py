"""
Eye Tracking & Blink Detection Module
======================================
Detects eye state, blinks, drowsiness, and gaze direction
using MediaPipe Face Mesh (468/478 landmarks).

Core Components:
    - EARCalculator      : Eye Aspect Ratio computation
    - BlinkDetector      : Blink counting & duration tracking
    - DrowsinessDetector : Sleepy/drowsy classification
    - GazeEstimator      : Iris-based gaze direction
    - EyeVisualizer      : Landmark + metric overlay drawing
    - EyeTrackingPipeline: End-to-end orchestrator

Usage:
    from eye_tracking import EyeTrackingPipeline
    pipeline = EyeTrackingPipeline()
    pipeline.run()   # live webcam
"""
from .ear_calculator import EARCalculator
from .blink_detector import BlinkDetector
from .drowsiness_detector import DrowsinessDetector
from .gaze_estimator import GazeEstimator
from .visualizer import EyeVisualizer
from .pipeline import EyeTrackingPipeline

__all__ = [
    "EARCalculator",
    "BlinkDetector",
    "DrowsinessDetector",
    "GazeEstimator",
    "EyeVisualizer",
    "EyeTrackingPipeline",
]
__version__ = "1.0.0"
