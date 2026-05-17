
"""
face_detection/annotator.py
Draws bounding boxes, IDs, confidence scores, and optional
landmarks on the video frame.
"""
import cv2
import numpy as np
from typing import Dict
from .config import cfg
from .tracker import TrackedStudent

# Colour palette — cycles for many students
COLORS = [
    (0, 200, 0),  # Green
    (255, 100, 0),  # Blue
    (0, 165, 255),  # Orange
    (200, 0, 200),  # Purple
    (0, 200, 255),  # Yellow
    (255, 0, 100),  # Pink-blue
    (100, 255, 100),  # Light green
    (255, 200, 0),  # Cyan
    (0, 100, 255),  # Red-orange
    (200, 200, 0),  # Teal
]


class FrameAnnotator:
    """
    Draws detection results onto frames.

    Usage:
        annotator = FrameAnnotator()
        frame = annotator.draw(frame, tracked_students, fps=30.0)
    """

    def draw(self, frame: np.ndarray, students: Dict[int, TrackedStudent],
             fps: float = 0.0) -> np.ndarray:
        """
        Annotate frame with all tracked students.

        Args:
            frame: BGR frame to draw on.
            students: Dict from CentroidTracker.
            fps: Current FPS to display.

        Returns:
            Annotated frame (same object, modified in-place).
        """
        for sid, student in students.items():
            if student.disappeared > 0:
                continue  # Don't draw disappeared faces

            color = COLORS[sid % len(COLORS)]
            x1, y1, x2, y2 = student.bbox

            # ---- Bounding Box ----
            # Rounded corner rectangle for modern look
            self._rounded_rect(frame, (x1, y1), (x2, y2), color,
                               thickness=cfg.bbox_thickness, radius=12)

            # ---- ID Label ----
            label = f"Student {sid}"
            if cfg.show_confidence:
                label += f" ({student.confidence:.0%})"

            # Label background
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, 1)
            cv2.rectangle(frame,
                          (x1, y1 - th - 14),
                          (x1 + tw + 12, y1),
                          color, -1)
            cv2.putText(frame, label, (x1 + 6, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # ---- Center Dot ----
            cx, cy = student.centroid
            cv2.circle(frame, (cx, cy), 4, color, -1)

            # ---- Optional: Mesh Landmarks ----
            if cfg.show_landmarks and student.landmarks_478:
                h, w = frame.shape[:2]
                for lm in student.landmarks_478:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    cv2.circle(frame, (px, py), 1, (200, 200, 200), -1)

        # ---- FPS Counter ----
        if cfg.show_fps and fps > 0:
            fps_text = f"FPS: {fps:.1f}"
            cv2.rectangle(frame, (8, 8), (140, 40), (0, 0, 0), -1)
            cv2.putText(frame, fps_text, (14, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)

        # ---- Student Count ----
        active = sum(1 for s in students.values() if s.disappeared == 0)
        count_text = f"Students: {active}"
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w - 170, 8), (w - 8, 40), (0, 0, 0), -1)
        cv2.putText(frame, count_text, (w - 164, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2, cv2.LINE_AA)

        return frame

    @staticmethod
    def _rounded_rect(img, pt1, pt2, color, thickness=2, radius=10):
        """Draw a rounded rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        r = min(radius, (x2 - x1) // 4, (y2 - y1) // 4)

        # Straight edges
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)

        # Corners
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

