
"""
Run: python -m head_pose
"""
from .pipeline import HeadPosePipeline

if __name__ == "__main__":
    pipeline = HeadPosePipeline(camera_index=0)
    pipeline.run()
