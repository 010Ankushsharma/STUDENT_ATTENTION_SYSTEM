
"""
face_detection/tracker.py
Centroid Tracker — assigns persistent integer IDs to faces
across frames using the Hungarian algorithm (scipy).

How it works:
  1. New face detected → register with next available ID
  2. Existing face moves → match to closest previous centroid
  3. Face disappears → keep ID alive for N frames, then deregister
  4. Face reappears nearby → reuse the same ID

This avoids expensive re-identification and works well for
classroom scenarios where students don't move much.
"""
import numpy as np
from scipy.spatial.distance import cdist
from collections import OrderedDict
from typing import List, Tuple, Dict
from .config import cfg
from .detector import DetectedFace


class TrackedStudent:
    """Represents a tracked student with persistent ID."""

    __slots__ = ("student_id", "centroid", "bbox", "confidence",
                 "landmarks_478", "disappeared", "frame_count")

    def __init__(self, student_id: int, centroid: Tuple[int, int],
                 bbox: Tuple[int, int, int, int], confidence: float,
                 landmarks_478=None):
        self.student_id = student_id
        self.centroid = centroid
        self.bbox = bbox
        self.confidence = confidence
        self.landmarks_478 = landmarks_478
        self.disappeared = 0
        self.frame_count = 1

    def __repr__(self):
        return f"Student(id={self.student_id}, pos={self.centroid}, frames={self.frame_count})"


class CentroidTracker:
    """
    Assigns and maintains persistent student IDs across video frames.

    Usage:
        tracker = CentroidTracker()

        # Each frame:
        faces = detector.detect(frame)
        students = tracker.update(faces)

        for student in students.values():
            print(f"Student {student.student_id} at {student.bbox}")
    """

    def __init__(self, max_disappeared: int = None, max_distance: int = None):
        self._max_disappeared = max_disappeared or cfg.max_disappeared
        self._max_distance = max_distance or cfg.max_distance
        self._next_id = 0
        self._students: OrderedDict[int, TrackedStudent] = OrderedDict()

    # ---- Public API ----

    @property
    def students(self) -> OrderedDict:
        """Current tracked students {id: TrackedStudent}."""
        return self._students

    @property
    def active_count(self) -> int:
        """Number of currently visible students."""
        return sum(1 for s in self._students.values() if s.disappeared == 0)

    def update(self, faces: List[DetectedFace]) -> OrderedDict:
        """
        Update tracker with new detections.

        Args:
            faces: List of DetectedFace from the current frame.

        Returns:
            OrderedDict of {student_id: TrackedStudent}
        """
        # No detections → increment disappeared for all
        if len(faces) == 0:
            for sid in list(self._students.keys()):
                self._students[sid].disappeared += 1
                if self._students[sid].disappeared > self._max_disappeared:
                    self._deregister(sid)
            return self._students

        # Extract centroids from new detections
        new_centroids = np.array([f.center for f in faces])

        # No existing students → register all
        if len(self._students) == 0:
            for i, face in enumerate(faces):
                self._register(face)
            return self._students

        # Match existing students to new detections
        existing_ids = list(self._students.keys())
        existing_centroids = np.array([
            self._students[sid].centroid for sid in existing_ids
        ])

        # Compute distance matrix
        D = cdist(existing_centroids, new_centroids)

        # Hungarian-style greedy matching (rows = existing, cols = new)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # Distance too large → not a match
            if D[row, col] > self._max_distance:
                continue

            # Update matched student
            sid = existing_ids[row]
            self._students[sid].centroid = faces[col].center
            self._students[sid].bbox = faces[col].bbox
            self._students[sid].confidence = faces[col].confidence
            self._students[sid].landmarks_478 = faces[col].landmarks_478
            self._students[sid].disappeared = 0
            self._students[sid].frame_count += 1

            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing → mark disappeared
        for row in range(len(existing_ids)):
            if row not in used_rows:
                sid = existing_ids[row]
                self._students[sid].disappeared += 1
                if self._students[sid].disappeared > self._max_disappeared:
                    self._deregister(sid)

        # Handle unmatched new → register
        for col in range(len(faces)):
            if col not in used_cols:
                self._register(faces[col])

        return self._students

    def reset(self):
        """Clear all tracked students."""
        self._students.clear()
        self._next_id = 0

    # ---- Internal ----

    def _register(self, face: DetectedFace):
        self._students[self._next_id] = TrackedStudent(
            student_id=self._next_id,
            centroid=face.center,
            bbox=face.bbox,
            confidence=face.confidence,
            landmarks_478=face.landmarks_478,
        )
        self._next_id += 1

    def _deregister(self, student_id: int):
        del self._students[student_id]