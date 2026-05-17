
"""
head_pose/visualizer.py
Draws head pose overlays: 3D axes, direction labels, angle gauges,
wireframe, and attention zone indicators.
"""
import cv2
import numpy as np
import math
from typing import Optional
from .config import cfg
from .pose_calculator import PoseResult
from .direction_classifier import DirectionResult
from .attention_judge import AttentionResult

# Colours
RED = (0, 0, 255)
GREEN = (0, 220, 0)
BLUE = (255, 100, 0)
YELLOW = (0, 220, 255)
ORANGE = (0, 140, 255)
CYAN = (255, 200, 0)
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)
DARK = (30, 30, 30)

ZONE_COLORS = {
    "attentive": GREEN,
    "marginal": YELLOW,
    "inattentive": RED,
}


class PoseVisualizer:
    """
    Draws pose-related overlays on the video frame.

    Features:
        - 3D RGB axes on nose (X=red, Y=green, Z=blue)
        - Direction label above head
        - Yaw/Pitch/Roll gauge arcs
        - Attention zone badge
        - Face wireframe (optional)
        - Landmark points (optional)
    """

    def draw(self, frame: np.ndarray,
             pose: Optional[PoseResult] = None,
             direction: Optional[DirectionResult] = None,
             attention: Optional[AttentionResult] = None,
             face_contour: Optional[np.ndarray] = None,
             mapped_points: Optional[np.ndarray] = None,
             face_index: int = 0) -> np.ndarray:
        """
        Annotate frame with head pose visualisations.

        Args:
            frame: BGR frame.
            pose: PoseResult from PoseCalculator.
            direction: DirectionResult from DirectionClassifier.
            attention: AttentionResult from AttentionJudge.
            face_contour: Face oval points for wireframe.
            mapped_points: Raw 2D landmark points.
            face_index: Index for multi-face labelling.

        Returns:
            Annotated frame.
        """
        if pose is None:
            return frame

        # 1. 3D Axes
        if cfg.draw_3d_axes and pose.axis_points_2d is not None:
            self._draw_axes(frame, pose.axis_points_2d, pose.nose_tip_2d)

        # 2. Direction label
        if cfg.draw_direction_label and direction:
            self._draw_direction_label(frame, pose.nose_tip_2d, direction, face_index)

        # 3. Angle gauges
        if cfg.draw_angle_gauges:
            self._draw_gauges(frame, pose, face_index)

        # 4. Attention badge
        if attention:
            self._draw_attention_badge(frame, attention, pose.nose_tip_2d, face_index)

        # 5. Wireframe
        if cfg.draw_face_wireframe and face_contour is not None:
            cv2.polylines(frame, [face_contour], True, GRAY, 1, cv2.LINE_AA)

        # 6. Landmark points
        if cfg.draw_landmark_points and mapped_points is not None:
            for i, (px, py) in enumerate(mapped_points.astype(int)):
                cv2.circle(frame, (px, py), 3, CYAN, -1)
                cv2.putText(frame, str(i), (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, CYAN, 1)

        return frame

    # ──────────── Drawing Helpers ────────────

    def _draw_axes(self, frame, axis_pts, nose_tip):
        """Draw RGB 3D axes from nose tip."""
        origin = tuple(axis_pts[0])
        x_end = tuple(axis_pts[1])
        y_end = tuple(axis_pts[2])
        z_end = tuple(axis_pts[3])

        cv2.line(frame, origin, x_end, (0, 0, 255), 3)  # X = Red
        cv2.line(frame, origin, y_end, (0, 255, 0), 3)  # Y = Green
        cv2.line(frame, origin, z_end, (255, 0, 0), 3)  # Z = Blue

        # Axis labels
        cv2.putText(frame, "X", x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1)
        cv2.putText(frame, "Y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
        cv2.putText(frame, "Z", z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLUE, 1)

    def _draw_direction_label(self, frame, nose_tip, direction, face_idx):
        """Draw direction label above the head."""
        label = direction.combined_label.upper()
        severity = direction.severity

        if direction.is_forward:
            color = GREEN
        elif severity in ("slight", "none"):
            color = YELLOW
        elif severity == "moderate":
            color = ORANGE
        else:
            color = RED

        # Tilted indicator
        tilt_icon = " ⟲" if direction.is_tilted else ""

        text = f"F{face_idx}: {label}{tilt_icon}"
        x = nose_tip[0] - 60
        y = nose_tip[1] - 80

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 8, y + 6), DARK, -1)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    def _draw_gauges(self, frame, pose, face_idx):
        """Draw compact yaw/pitch/roll arc gauges."""
        h, w = frame.shape[:2]
        base_x = 15
        base_y = 50 + face_idx * 150

        gauges = [
            ("YAW", pose.yaw, 60, RED),
            ("PITCH", pose.pitch, 60, GREEN),
            ("ROLL", pose.roll, 60, BLUE),
        ]

        for i, (name, angle, max_angle, color) in enumerate(gauges):
            cx = base_x + 40
            cy = base_y + i * 50
            radius = 22

            # Background arc
            cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -135, 135, GRAY, 2)

            # Value arc
            angle_clamped = max(-max_angle, min(max_angle, angle))
            end_angle = -90 + (angle_clamped / max_angle) * 135
            start = -90
            cv2.ellipse(frame, (cx, cy), (radius, radius), 0,
                        min(start, end_angle), max(start, end_angle), color, 3)

            # Needle
            theta = math.radians(-90 + (angle_clamped / max_angle) * 135)
            nx = int(cx + radius * math.cos(theta))
            ny = int(cy + radius * math.sin(theta))
            cv2.line(frame, (cx, cy), (nx, ny), WHITE, 2)

            # Labels
            cv2.putText(frame, name, (cx + radius + 8, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
            cv2.putText(frame, f"{angle:.1f} deg", (cx + radius + 8, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, WHITE, 1)

    def _draw_attention_badge(self, frame, attention, nose_tip, face_idx):
        """Draw attention zone badge below the face."""
        zone = attention.zone
        color = ZONE_COLORS.get(zone, GRAY)
        score_pct = int(attention.attention_score * 100)

        text = f"{zone.upper()} ({score_pct}%)"
        x = nose_tip[0] - 70
        y = nose_tip[1] + 40

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 8, y + 6), color, -1)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

        # Short description
        cv2.putText(frame, attention.description, (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)