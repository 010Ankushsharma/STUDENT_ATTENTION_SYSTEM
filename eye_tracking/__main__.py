
"""
Run: python -m eye_tracking
"""
from .pipeline import EyeTrackingPipeline

if __name__ == "__main__":
    pipeline = EyeTrackingPipeline(camera_index=0)
    pipeline.run()
