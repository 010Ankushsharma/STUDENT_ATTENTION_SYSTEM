"""
Face Detection Module for Student Attention Detection System
=============================================================
Detects multiple faces in real-time using MediaPipe + OpenCV,
assigns persistent IDs via centroid tracking, and draws annotated
bounding boxes.

Usage:
    from face_detection import FaceDetectionPipeline
    pipeline = FaceDetectionPipeline()
    pipeline.run()
"""
from .detector import FaceDetector
from .tracker import CentroidTracker
from .pipeline import FaceDetectionPipeline

__all__ = ["FaceDetector", "CentroidTracker", "FaceDetectionPipeline"]
__version__ = "1.0.0"
