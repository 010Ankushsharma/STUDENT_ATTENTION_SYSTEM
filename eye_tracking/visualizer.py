
"""
eye_tracking/visualizer.py
Draws eye tracking landmarks, EAR bars, metrics panel, and gaze arrows.
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from .config import cfg
from .blink_detector import BlinkMetrics
from .drowsiness_detector import DrowsinessResult, DrowsinessLevel
from .gaze_estimator import GazeResult

# ── Colour palette ──
GREEN = (0, 200, 0)
YELLOW = (0, 220, 255)
ORANGE = (0, 140, 255)
RED = (0, 0, 255)
CYAN = (255, 200, 0)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
DARK = (30, 30, 30)

DROWSY_COLORS = {
    DrowsinessLevel.ALERT: GREEN,
    DrowsinessLevel.MILD: YELLOW,
    DrowsinessLevel.MODERATE: ORANGE,
    DrowsinessLevel.SEVERE: RED,
}


class EyeVisualizer:
    """
    Draws eye tracking visualisations on the video frame.

    Features:
        - Eye contour outlines (left & right)
        - Iris circle markers
        - EAR progress bar
        - Metrics panel (blinks, rate, PERCLOS, drowsiness)
        - Gaze direction arrow
        - Drowsiness status badge
    """

    def draw(self, frame: np.ndarray, landmarks,
             frame_w: int, frame_h: int,
             blink_metrics: Optional[BlinkMetrics] = None,
             drowsiness: Optional[DrowsinessResult] = None,
             gaze: Optional[GazeResult] = None) -> np.ndarray:
        """
        Draw all visualisations on frame.

        Args:
            frame:          BGR frame to annotate.
            landmarks:      MediaPipe face landmarks.
            frame_w, frame_h: Frame dimensions.
            blink_metrics:  From BlinkDetector.
            drowsiness:     From DrowsinessDetector.
            gaze:           From GazeEstimator.

        Returns:
            Annotated frame.
        """
        if landmarks is None:
            return frame

        # 1. Eye contours
        if cfg.draw_eye_contours:
            self._draw_eye_contour(frame, landmarks, cfg.left_eye_contour_idx,
                                   frame_w, frame_h, CYAN)
            self._draw_eye_contour(frame, landmarks, cfg.right_eye_contour_idx,
                                   frame_w, frame_h, CYAN)

        # 2. Iris markers
        if cfg.draw_iris and cfg.refine_landmarks:
            self._draw_iris(frame, landmarks, cfg.left_iris_idx, frame_w, frame_h)
            self._draw_iris(frame, landmarks, cfg.right_iris_idx, frame_w, frame_h)

        # 3. EAR bar
        if cfg.draw_ear_bar and blink_metrics:
            self._draw_ear_bar(frame, blink_metrics.current_ear, frame_w, frame_h)

        # 4. Gaze arrow
        if cfg.show_gaze_arrow and gaze:
            self._draw_gaze_arrow(frame, landmarks, gaze, frame_w, frame_h)

        # 5. Metrics panel
        if cfg.draw_metrics_panel and blink_metrics:
            self._draw_metrics_panel(frame, blink_metrics, drowsiness, gaze)

        return frame

    # ──────────── Drawing Helpers ────────────

    def _draw_eye_contour(self, frame, landmarks, indices,
                          w, h, color, thickness=1):
        """Draw the eye outline polygon."""
        pts = np.array([
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in indices
        ], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

    def _draw_iris(self, frame, landmarks, iris_indices, w, h):
        """Draw iris center and circle."""
        if len(iris_indices) < 5:
            return
        center_idx = iris_indices[0]
        cx = int(landmarks[center_idx].x * w)
        cy = int(landmarks[center_idx].y * h)

        # Radius from cardinal points
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
               for i in iris_indices[1:]]
        radii = [np.sqrt((px - cx) ** 2 + (py - cy) ** 2) for px, py in pts]
        radius = int(np.mean(radii))

        cv2.circle(frame, (cx, cy), radius, GREEN, 1)
        cv2.circle(frame, (cx, cy), 2, GREEN, -1)

    def _draw_ear_bar(self, frame, ear, fw, fh):
        """Draw a vertical EAR progress bar on the left side."""
        bar_x = 20
        bar_y = 60
        bar_w = 20
        bar_h = 150

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      DARK, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      GRAY, 1)

        # Fill level (EAR 0.0-0.4 mapped to bar height)
        fill = max(0, min(1, ear / 0.4))
        fill_h = int(bar_h * fill)
        fill_y = bar_y + bar_h - fill_h

        # Color based on level
        if ear >= cfg.ear_partial_threshold:
            color = GREEN
        elif ear >= cfg.ear_blink_threshold:
            color = YELLOW
        else:
            color = RED

        cv2.rectangle(frame, (bar_x + 2, fill_y),
                      (bar_x + bar_w - 2, bar_y + bar_h - 2), color, -1)

        # Threshold line
        thresh_y = bar_y + bar_h - int(bar_h * (cfg.ear_blink_threshold / 0.4))
        cv2.line(frame, (bar_x - 4, thresh_y), (bar_x + bar_w + 4, thresh_y),
                 RED, 1)

        # Labels
        cv2.putText(frame, "EAR", (bar_x - 2, bar_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        cv2.putText(frame, f"{ear:.3f}", (bar_x - 5, bar_y + bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)

    def _draw_gaze_arrow(self, frame, landmarks, gaze, w, h):
        """Draw gaze direction arrow between the eyes."""
        # Midpoint between eyes
        le = landmarks[468] if len(landmarks) > 468 else landmarks[159]
        re = landmarks[473] if len(landmarks) > 473 else landmarks[386]
        mid_x = int((le.x + re.x) / 2 * w)
        mid_y = int((le.y + re.y) / 2 * h)

        # Arrow direction from ratios
        dx = int((gaze.h_ratio - 0.5) * 80)
        dy = int((gaze.v_ratio - 0.5) * 60)

        color = RED if gaze.is_looking_away else GREEN
        cv2.arrowedLine(frame, (mid_x, mid_y), (mid_x + dx, mid_y + dy),
                        color, 2, tipLength=0.35)

    def _draw_metrics_panel(self, frame, blink_metrics, drowsiness, gaze):
        """Draw translucent metrics panel in top-right."""
        h, w = frame.shape[:2]
        panel_w = 260
        panel_h = 200
        px = w - panel_w - 10
        py = 10

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), DARK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), GRAY, 1)

        # Title
        cv2.putText(frame, "EYE METRICS", (px + 10, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)

        # Metrics lines
        y = py + 45
        line_h = 22
        metrics_lines = [
            (f"EAR: {blink_metrics.current_ear:.3f}", WHITE),
            (f"Blinks: {blink_metrics.total_blinks}", WHITE),
            (f"Rate: {blink_metrics.blink_rate_per_min:.1f} /min", WHITE),
            (f"PERCLOS: {blink_metrics.perclos:.1%}",
             RED if blink_metrics.perclos > cfg.perclos_threshold else WHITE),
            (f"Avg Duration: {blink_metrics.avg_blink_duration_ms:.0f} ms", WHITE),
        ]

        if drowsiness:
            color = DROWSY_COLORS.get(drowsiness.level, WHITE)
            metrics_lines.append(
                (f"Drowsiness: {drowsiness.level.value.upper()}", color)
            )

        if gaze:
            g_color = RED if gaze.is_looking_away else GREEN
            metrics_lines.append(
                (f"Gaze: {gaze.direction_label}", g_color)
            )

        for text, color in metrics_lines:
            cv2.putText(frame, text, (px + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            y += line_h

        # Eyes status badge
        if blink_metrics.eyes_closed:
            badge_color = RED
            badge_text = "EYES CLOSED"
        else:
            badge_color = GREEN
            badge_text = "EYES OPEN"

        bx = px + 10
        by = py + panel_h - 30
        cv2.rectangle(frame, (bx, by), (bx + 120, by + 22), badge_color, -1)
        cv2.putText(frame, badge_text, (bx + 5, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)