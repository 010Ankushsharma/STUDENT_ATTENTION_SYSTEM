"""
clean_overlay.py
════════════════════════════════════════════════════════════════════════
DROP-IN REPLACEMENT for the overlay drawing code in main_app.py.

WHAT THIS REMOVES (the visual noise):
  ✗ EAR vertical bar (left side)
  ✗ Yaw / pitch dial / gauge
  ✗ "F1: DOWN-LEFT" giant red direction text
  ✗ "EYES OPEN" / "EYES CLOSED" green/red box
  ✗ "Drowsiness: ALERT" floating label
  ✗ "EYE METRICS" side panel
  ✗ Blink duration text
  ✗ Sub-module visualizer overlays (eye contours, pose axes)

WHAT THIS KEEPS:
  ✓ Coloured bounding box per student — colour = state
  ✓ Small chip above box:  "S1  87%"
  ✓ 4px score bar directly under the box
  ✓ Slim bottom bar: state counts + class avg bar + FPS + frame
  ✓ Red border flash on alert (unchanged in main_app.py)

HOW TO INTEGRATE  (3 small edits to main_app.py)
════════════════════════════════════════════════════════════════════════

EDIT 1 – swap the import at the top of main_app.py
────────────────────────────────────────────────────
Remove or comment out DashboardDrawer, then add:

    from clean_overlay import CleanOverlay

In StudentAttentionSystem.__init__() change:
    self.dashboard_drawer = DashboardDrawer()
to:
    self.dashboard_drawer = CleanOverlay()

EDIT 2 – silence the eye visualizer inside _process_student()
──────────────────────────────────────────────────────────────
Find and DELETE (or comment out) this block:

    frame = proc.eye_vis.draw(frame, lm, pw, ph,
                              blink_metrics=blink_metrics,
                              drowsiness=drowsy_result,
                              gaze=gaze_result)

EDIT 3 – silence the pose visualizer inside _process_student()
───────────────────────────────────────────────────────────────
Find and DELETE (or comment out) this block:

    frame = proc.pose_vis.draw(frame, pose, direction,
                               attention, face_index=sid)

Nothing else needs changing.  The two surviving calls:
    self.dashboard_drawer.draw(...)           # in process_frame()
    self.dashboard_drawer.draw_student_badge(...) # in _process_student()
have identical signatures to DashboardDrawer so they work unchanged.
════════════════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np


# ── Palette (BGR — OpenCV order) ───────────────────────────────────────
_STATE_COLOR = {
    "attentive":    ( 80, 200,  80),   # green
    "distracted":   ( 40, 190, 220),   # yellow-cyan
    "sleepy":       ( 60,  60, 220),   # red
    "looking_away": (200, 100,  40),   # blue-indigo
}
_DEFAULT_COLOR = (120, 120, 120)
_WHITE     = (240, 240, 240)
_DARK      = ( 18,  20,  24)
_GRAY      = ( 80,  84,  92)


def _color(state: str) -> tuple:
    return _STATE_COLOR.get(state, _DEFAULT_COLOR)


def _alpha_rect(frame, x1, y1, x2, y2, bgr, alpha: float = 0.60):
    """Semi-transparent filled rectangle."""
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    overlay = np.full_like(roi, bgr, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi


def _put(frame, text, x, y, scale=0.36, color=_WHITE, thickness=1):
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA,
    )


def _text_size(text, scale=0.36, thickness=1):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return w, h


# ══════════════════════════════════════════════════════════════════════
class CleanOverlay:
    """
    Minimal HUD — drop-in for DashboardDrawer.

    Public API (identical signatures to DashboardDrawer):
        draw_student_badge(frame, bbox, student_id, score, state)
        draw(frame, scoring, fps, frame_count)
    """

    # ── per-student ───────────────────────────────────────────────
    @staticmethod
    def draw_student_badge(frame, bbox, student_id, score, state):
        """
        Draws three things only:
          1. Coloured bounding box (2 px stroke)
          2. Small chip above the box: "S1  87%"  on a semi-transparent bg
          3. Slim 4 px score bar immediately below the box
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        col   = _color(state)
        label = f"S{student_id}  {int(score * 100)}%"

        # 1 · bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

        # 2 · label chip above the box
        PAD  = 5
        tw, th = _text_size(label, 0.36)
        cy1  = max(0, y1 - th - PAD * 2)
        cy2  = y1
        _alpha_rect(frame, x1, cy1, x1 + tw + PAD * 2, cy2, col, alpha=0.75)
        _put(frame, label, x1 + PAD, cy2 - PAD, 0.36, _WHITE, 1)

        # 3 · score bar below the box
        BAR_H = 4
        bar_y = min(y2 + 2, frame.shape[0] - BAR_H - 1)
        bar_w = x2 - x1
        _alpha_rect(frame, x1, bar_y, x2, bar_y + BAR_H, (30, 30, 30), alpha=0.85)
        fill_w = max(2, int(bar_w * max(0.0, min(1.0, score))))
        cv2.rectangle(frame, (x1, bar_y), (x1 + fill_w, bar_y + BAR_H), col, -1)

        return frame

    # ── classroom bar ─────────────────────────────────────────────
    @staticmethod
    def draw(frame, scoring, fps: float, frame_count: int):
        """
        Single slim bar at the bottom of the frame:
          Left    four coloured dots + state counts
          Centre  class-average filled bar + % label
          Right   fps · frame number
        """
        h, w = frame.shape[:2]
        BAR_H = 38
        y0    = h - BAR_H

        # background
        _alpha_rect(frame, 0, y0, w, h, _DARK, alpha=0.82)
        cv2.line(frame, (0, y0), (w, y0), (52, 56, 64), 1)

        summary = scoring.get_class_summary()

        # ── state dots + counts (left) ────────────────────────────
        _STATES = [
            ("attentive",    "Attn"),
            ("distracted",   "Dist"),
            ("sleepy",       "Slpy"),
            ("looking_away", "Away"),
        ]
        DOT_R = 4
        x     = 14
        dot_y = y0 + 14
        txt_y = y0 + 28

        for key, short in _STATES:
            col   = _color(key)
            count = summary.get(key, 0)
            cv2.circle(frame, (x + DOT_R, dot_y), DOT_R, col, -1, cv2.LINE_AA)
            seg = f"{short} {count}"
            _put(frame, seg, x, txt_y, 0.32, (155, 160, 170), 1)
            tw, _ = _text_size(seg, 0.32)
            x += tw + 20

        # ── class avg bar (centre) ────────────────────────────────
        avg    = float(summary.get("avg_score", 0.0))
        bx1    = w // 3
        bx2    = (w * 2) // 3
        bar_y  = y0 + 13
        BAR_FH = 7

        # track
        cv2.rectangle(frame, (bx1, bar_y), (bx2, bar_y + BAR_FH), (44, 48, 56), -1)
        # fill
        fill_col = (
            _color("attentive")  if avg >= 0.70 else
            _color("distracted") if avg >= 0.45 else
            _color("sleepy")
        )
        fill_x2 = bx1 + max(2, int((bx2 - bx1) * avg))
        cv2.rectangle(frame, (bx1, bar_y), (fill_x2, bar_y + BAR_FH), fill_col, -1)

        # label centred under bar
        avg_lbl = f"Class avg  {int(avg * 100)}%"
        tw, _ = _text_size(avg_lbl, 0.32)
        cx = bx1 + ((bx2 - bx1) - tw) // 2
        _put(frame, avg_lbl, cx, y0 + 32, 0.32, (155, 160, 170), 1)

        # ── fps + frame (right) ───────────────────────────────────
        meta = f"{fps:.1f} fps   #{frame_count}"
        tw, _ = _text_size(meta, 0.32)
        _put(frame, meta, w - tw - 12, y0 + 28, 0.32, _GRAY, 1)

        return frame