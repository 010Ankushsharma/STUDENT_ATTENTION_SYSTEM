
"""
face_detection/detector.py
Core face detection using MediaPipe Face Mesh (primary) with
Face Detection (fallback).

Why Face Mesh over Face Detection?
  - Returns 468 landmarks per face (needed later for EAR, head pose)
  - Built-in multi-face support
  - Better temporal stability (tracking confidence)

The detector returns a list of DetectedFace dataclass objects.
"""
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .config import cfg


@dataclass
class DetectedFace:
    """Single detected face with all metadata."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]  # (cx, cy)
    confidence: float = 1.0
    landmarks_478: Optional[list] = None  # Full mesh landmarks (raw)
    face_width: int = 0
    face_height: int = 0

    @property
    def area(self) -> int:
        return self.face_width * self.face_height


class FaceDetector:
    """
    Detects faces using MediaPipe.

    Two modes:
      - Face Mesh mode (default): richer landmarks, better tracking
      - Face Detection mode: lighter, faster for simple bbox needs

    Usage:
        detector = FaceDetector()
        faces = detector.detect(frame)
        for face in faces:
            print(face.bbox, face.confidence)
        detector.release()
    """

    def __init__(self, use_mesh: bool = None):
        self._use_mesh = use_mesh if use_mesh is not None else cfg.use_face_mesh

        if self._use_mesh:
            self._mp_mesh = mp.solutions.face_mesh
            self._model = self._mp_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=cfg.max_num_faces,
                refine_landmarks=cfg.mesh_refine_landmarks,
                min_detection_confidence=cfg.mesh_min_detection,
                min_tracking_confidence=cfg.mesh_min_tracking,
            )
        else:
            self._mp_det = mp.solutions.face_detection
            self._model = self._mp_det.FaceDetection(
                model_selection=cfg.model_selection,
                min_detection_confidence=cfg.min_detection_confidence,
            )

    # ---- Public API ----

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        Detect all faces in a BGR frame.

        Args:
            frame: BGR numpy array from OpenCV.

        Returns:
            List of DetectedFace objects.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe expects writeable=False for performance
        rgb.flags.writeable = False
        results = self._model.process(rgb)
        rgb.flags.writeable = True

        if self._use_mesh:
            return self._parse_mesh_results(results, w, h)
        else:
            return self._parse_detection_results(results, w, h)

    def release(self):
        """Free MediaPipe resources."""
        self._model.close()

    # ---- Parsing ----

    def _parse_mesh_results(self, results, w: int, h: int) -> List[DetectedFace]:
        """Extract faces from Face Mesh results."""
        faces: List[DetectedFace] = []

        if not results.multi_face_landmarks:
            return faces

        for face_lm in results.multi_face_landmarks:
            lm = face_lm.landmark

            # Compute tight bounding box from all 468+ landmarks
            xs = [l.x * w for l in lm]
            ys = [l.y * h for l in lm]

            x1 = max(0, int(min(xs)) - 10)  # 10px padding
            y1 = max(0, int(min(ys)) - 10)
            x2 = min(w, int(max(xs)) + 10)
            y2 = min(h, int(max(ys)) + 10)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Confidence from nose tip visibility
            conf = round(getattr(lm[1], 'visibility', 1.0), 2)

            faces.append(DetectedFace(
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                confidence=conf,
                landmarks_478=[lm[i] for i in range(len(lm))],
                face_width=x2 - x1,
                face_height=y2 - y1,
            ))

        return faces

    def _parse_detection_results(self, results, w: int, h: int) -> List[DetectedFace]:
        """Extract faces from Face Detection results."""
        faces: List[DetectedFace] = []

        if not results.detections:
            return faces

        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            bw = int(bb.width * w)
            bh = int(bb.height * h)
            x2 = min(w, x1 + bw)
            y2 = min(h, y1 + bh)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            faces.append(DetectedFace(
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                confidence=round(det.score[0], 2),
                face_width=bw,
                face_height=bh,
            ))

        return faces

