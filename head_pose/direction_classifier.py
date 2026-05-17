
"""
head_pose/direction_classifier.py
Converts raw Euler angles (yaw, pitch, roll) into human-readable
direction labels with granularity levels.

Direction Grid:
                        PITCH
              UP         ↑         UP
             LEFT        |        RIGHT
               ╲         |         ╱
                ╲   SLIGHT-UP    ╱
    ◄── LEFT ────── FORWARD ──────── RIGHT ──►  YAW
                ╱   SLIGHT-DOWN  ╲
               ╱         |         ╲
             DOWN        |        DOWN
             LEFT        ↓        RIGHT
                        DOWN

Classification Priority:
    1. Strong direction (beyond slight_range)  → "left", "right", "down", "up"
    2. Slight direction (between forward & slight) → "slight_left", etc.
    3. Within forward range → "forward"
"""
from dataclasses import dataclass
from typing import List
from .config import cfg
from .pose_calculator import PoseResult


@dataclass
class DirectionResult:
    """Classified head direction."""
    primary_direction: str  # "forward", "left", "right", "up", "down"
    horizontal: str  # "center", "slight_left", "left", "slight_right", "right"
    vertical: str  # "center", "slight_up", "up", "slight_down", "down"
    combined_label: str  # e.g. "forward", "down-left", "slight_right"
    is_forward: bool  # True if looking roughly ahead
    is_tilted: bool  # Head tilted sideways
    yaw: float
    pitch: float
    roll: float
    severity: str  # "none", "slight", "moderate", "strong"


class DirectionClassifier:
    """
    Classifies head direction from pose angles.

    Usage:
        classifier = DirectionClassifier()
        direction = classifier.classify(pose_result)
        print(direction.combined_label)  # e.g. "slight_left"
        print(direction.is_forward)      # True/False
    """

    def classify(self, pose: PoseResult) -> DirectionResult:
        """
        Classify the head direction from a PoseResult.

        Args:
            pose: PoseResult from PoseCalculator.

        Returns:
            DirectionResult with granular direction labels.
        """
        yaw = pose.yaw
        pitch = pose.pitch
        roll = pose.roll

        # ── Horizontal classification ──
        yf_lo, yf_hi = cfg.yaw_forward_range
        ys_lo, ys_hi = cfg.yaw_slight_range

        if yf_lo <= yaw <= yf_hi:
            horizontal = "center"
        elif ys_lo <= yaw < yf_lo:
            horizontal = "slight_left"
        elif yf_hi < yaw <= ys_hi:
            horizontal = "slight_right"
        elif yaw < ys_lo:
            horizontal = "left"
        else:
            horizontal = "right"

        # ── Vertical classification ──
        pf_lo, pf_hi = cfg.pitch_forward_range
        ps_lo, ps_hi = cfg.pitch_slight_range

        if pf_lo <= pitch <= pf_hi:
            vertical = "center"
        elif ps_lo <= pitch < pf_lo:
            vertical = "slight_down"
        elif pf_hi < pitch <= ps_hi:
            vertical = "slight_up"
        elif pitch < ps_lo:
            vertical = "down"
        else:
            vertical = "up"

        # ── Combined label ──
        is_forward = (horizontal == "center" and vertical == "center")
        is_tilted = abs(roll) > cfg.roll_alert_threshold

        if is_forward:
            combined = "forward"
            primary = "forward"
        elif horizontal == "center":
            combined = vertical
            primary = vertical.replace("slight_", "")
        elif vertical == "center":
            combined = horizontal
            primary = horizontal.replace("slight_", "")
        else:
            # Diagonal: pick dominant axis
            v_clean = vertical.replace("slight_", "")
            h_clean = horizontal.replace("slight_", "")
            combined = f"{v_clean}-{h_clean}"
            # Primary = whichever has larger deviation
            primary = h_clean if abs(yaw) > abs(pitch) else v_clean

        # ── Severity ──
        max_angle = max(abs(yaw), abs(pitch))
        if max_angle <= max(abs(yf_hi), abs(pf_hi)):
            severity = "none"
        elif max_angle <= max(abs(ys_hi), abs(ps_hi)):
            severity = "slight" if "slight" in combined else "moderate"
        else:
            severity = "strong"

        return DirectionResult(
            primary_direction=primary,
            horizontal=horizontal,
            vertical=vertical,
            combined_label=combined,
            is_forward=is_forward,
            is_tilted=is_tilted,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            severity=severity,
        )

