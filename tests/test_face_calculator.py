
"""
tests/test_face_detection.py
Unit tests for the face detection module.
Run: pytest tests/ -v
"""
import numpy as np
import pytest
from face_detection.detector import FaceDetector, DetectedFace
from face_detection.tracker import CentroidTracker, TrackedStudent
from face_detection.preprocessor import FramePreprocessor
from face_detection.config import cfg


class TestFaceDetector:
    """Tests for FaceDetector."""

    def test_init_mesh_mode(self):
        det = FaceDetector(use_mesh=True)
        assert det._use_mesh is True
        det.release()

    def test_init_detection_mode(self):
        det = FaceDetector(use_mesh=False)
        assert det._use_mesh is False
        det.release()

    def test_detect_blank_frame(self):
        """No faces in a blank frame."""
        det = FaceDetector(use_mesh=True)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = det.detect(blank)
        assert isinstance(faces, list)
        assert len(faces) == 0
        det.release()


class TestCentroidTracker:
    """Tests for CentroidTracker."""

    def _make_face(self, cx, cy, size=50):
        return DetectedFace(
            bbox=(cx - size, cy - size, cx + size, cy + size),
            center=(cx, cy),
            confidence=0.95,
            face_width=size * 2,
            face_height=size * 2,
        )

    def test_register_new_faces(self):
        tracker = CentroidTracker(max_disappeared=5, max_distance=80)
        faces = [self._make_face(100, 100), self._make_face(300, 100)]
        students = tracker.update(faces)
        assert len(students) == 2
        assert 0 in students
        assert 1 in students

    def test_persistent_ids(self):
        """Same face in same position keeps same ID."""
        tracker = CentroidTracker(max_disappeared=5, max_distance=80)

        # Frame 1
        tracker.update([self._make_face(100, 100)])
        # Frame 2 — same position
        students = tracker.update([self._make_face(105, 102)])

        assert len(students) == 1
        assert 0 in students  # Same ID

    def test_face_moves_keeps_id(self):
        tracker = CentroidTracker(max_disappeared=5, max_distance=80)
        tracker.update([self._make_face(100, 100)])
        # Move within threshold
        students = tracker.update([self._make_face(140, 120)])
        assert 0 in students

    def test_face_disappears_and_deregisters(self):
        tracker = CentroidTracker(max_disappeared=3, max_distance=80)
        tracker.update([self._make_face(100, 100)])
        # Disappear for 4 frames
        for _ in range(4):
            tracker.update([])
        assert len(tracker.students) == 0

    def test_multiple_faces_tracked(self):
        tracker = CentroidTracker(max_disappeared=5, max_distance=80)
        faces = [self._make_face(100, 100), self._make_face(300, 300)]
        tracker.update(faces)

        # Move both slightly
        faces2 = [self._make_face(110, 105), self._make_face(295, 305)]
        students = tracker.update(faces2)

        assert len(students) == 2
        assert students[0].centroid == (110, 105)
        assert students[1].centroid == (295, 305)

    def test_new_face_gets_new_id(self):
        tracker = CentroidTracker(max_disappeared=5, max_distance=80)
        tracker.update([self._make_face(100, 100)])
        # Add second face far away
        students = tracker.update([
            self._make_face(105, 102),
            self._make_face(500, 400),
        ])
        assert len(students) == 2
        ids = list(students.keys())
        assert 0 in ids  # Original
        assert 1 in ids  # New

    def test_reset(self):
        tracker = CentroidTracker()
        tracker.update([self._make_face(100, 100)])
        tracker.reset()
        assert len(tracker.students) == 0
        assert tracker._next_id == 0


class TestFramePreprocessor:
    """Tests for FramePreprocessor."""

    def test_resize_large_frame(self):
        pp = FramePreprocessor()
        large = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = pp.process(large)
        assert result.shape[1] == cfg.process_width

    def test_small_frame_unchanged_width(self):
        pp = FramePreprocessor()
        small = np.zeros((240, 320, 3), dtype=np.uint8)
        result = pp.process(small)
        assert result.shape[1] == 320  # Not resized up

    def test_clahe_output_shape(self):
        pp = FramePreprocessor()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pp.process(frame)
        assert result.shape == (480, 640, 3)


class TestDetectedFace:
    def test_area(self):
        face = DetectedFace(
            bbox=(10, 20, 110, 170),
            center=(60, 95),
            face_width=100,
            face_height=150,
        )
        assert face.area == 15000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
