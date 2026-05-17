
"""
Allows running: python -m face_detection
"""
from .pipeline import FaceDetectionPipeline

if __name__ == "__main__":
    pipeline = FaceDetectionPipeline(camera_index=0)
    pipeline.run()
