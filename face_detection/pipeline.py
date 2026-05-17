
"""
face_detection/pipeline.py
Complete Face Detection Pipeline — ties everything together.

Usage:
    # As standalone demo:
    python -m face_detection.pipeline

    # As importable module:
    from face_detection import FaceDetectionPipeline
    pipeline = FaceDetectionPipeline(camera_index=0)
    pipeline.run()                    # blocking loop with display
    # OR
    frame, students = pipeline.process_frame(raw_frame)  # single frame
"""
import cv2
import time
import numpy as np
from typing import Tuple, Dict, Optional, Callable
from collections import deque

from .config import cfg
from .preprocessor import FramePreprocessor
from .detector import FaceDetector, DetectedFace
from .tracker import CentroidTracker, TrackedStudent
from .annotator import FrameAnnotator


class FaceDetectionPipeline:
    """
    End-to-end face detection pipeline.

    Flow:
        Raw Frame → Preprocess → Detect → Track → Annotate → Output

    Can be used standalone (with display window) or integrated
    into a larger system (FastAPI, etc.) via process_frame().
    """

    def __init__(self, camera_index: int = 0, use_mesh: bool = True):
        self.camera_index = camera_index
        self.preprocessor = FramePreprocessor()
        self.detector = FaceDetector(use_mesh=use_mesh)
        self.tracker = CentroidTracker()
        self.annotator = FrameAnnotator()

        # FPS tracking
        self._fps_window = deque(maxlen=30)
        self._prev_time = time.perf_counter()

        # Frame skip counter
        self._frame_count = 0
        self._last_faces = []

    # ---- Single-frame API (for integration) ----

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[int, TrackedStudent]]:
        """
        Process a single frame through the full pipeline.

        Args:
            frame: Raw BGR frame from any source.

        Returns:
            (annotated_frame, {student_id: TrackedStudent})
        """
        # 1. Pre-process
        processed = self.preprocessor.process(frame)

        # 2. Frame skipping for performance
        self._frame_count += 1
        if cfg.skip_frames > 0 and (self._frame_count % (cfg.skip_frames + 1) != 0):
            faces = self._last_faces
        else:
            # 3. Detect
            faces = self.detector.detect(processed)
            self._last_faces = faces

        # 4. Track (assign IDs)
        students = self.tracker.update(faces)

        # 5. Calculate FPS
        now = time.perf_counter()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            self._fps_window.append(1.0 / dt)
        fps = sum(self._fps_window) / len(self._fps_window) if self._fps_window else 0

        # 6. Annotate
        annotated = self.annotator.draw(processed, students, fps=fps)

        return annotated, students

    # ---- Standalone webcam loop ----

    def run(self, on_frame: Optional[Callable] = None):
        """
        Run live webcam detection with display window.

        Args:
            on_frame: Optional callback(frame, students) called each frame.
                      Useful for logging, database writes, etc.

        Controls:
            q / ESC  → Quit
            s        → Save screenshot
            l        → Toggle landmarks
            f        → Toggle FPS display
            r        → Reset tracker IDs
        """
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("ERROR: Cannot open webcam. Check camera index.")
            return

        print("=" * 55)
        print("  Face Detection Module — Live")
        print("=" * 55)
        print(f"  Camera index : {self.camera_index}")
        print(f"  Resolution   : {int(cap.get(3))}x{int(cap.get(4))}")
        print(f"  Mode         : {'Face Mesh (468 pts)' if cfg.use_face_mesh else 'Face Detection'}")
        print(f"  Max faces    : {cfg.max_num_faces}")
        print(f"  CLAHE        : {'ON' if cfg.apply_clahe else 'OFF'}")
        print("-" * 55)
        print("  Keys: [q]uit  [s]creenshot  [l]andmarks  [r]eset")
        print("=" * 55)

        screenshot_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed. Exiting.")
                break

            # Process
            annotated, students = self.process_frame(frame)

            # Callback
            if on_frame:
                on_frame(annotated, students)

            # Display
            cv2.imshow("Student Face Detection", annotated)

            # Key handling
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):  # q or ESC
                break
            elif key == ord('s'):  # Screenshot
                path = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(path, annotated)
                print(f"  Screenshot saved: {path}")
                screenshot_count += 1
            elif key == ord('l'):  # Toggle landmarks
                cfg.show_landmarks = not cfg.show_landmarks
                print(f"  Landmarks: {'ON' if cfg.show_landmarks else 'OFF'}")
            elif key == ord('f'):  # Toggle FPS
                cfg.show_fps = not cfg.show_fps
            elif key == ord('r'):  # Reset tracker
                self.tracker.reset()
                print("  Tracker reset — IDs cleared")

        cap.release()
        cv2.destroyAllWindows()
        self.detector.release()
        print("\nShutdown complete.")

    # ---- Utilities ----

    def get_student_count(self) -> int:
        """Return number of currently visible students."""
        return self.tracker.active_count

    def get_student_data(self) -> Dict[int, dict]:
        """Return serialisable student data."""
        result = {}
        for sid, s in self.tracker.students.items():
            if s.disappeared == 0:
                result[sid] = {
                    "student_id": s.student_id,
                    "bbox": s.bbox,
                    "center": s.centroid,
                    "confidence": s.confidence,
                    "frames_tracked": s.frame_count,
                }
        return result


# ---- Run as standalone script ----
if __name__ == "__main__":
    pipeline = FaceDetectionPipeline(camera_index=0)
    pipeline.run()
