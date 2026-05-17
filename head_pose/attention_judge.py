
"""
head_pose/attention_judge.py
Determines if a student is looking toward the teacher/screen
based on head pose angles and configurable teacher position.

Attention Zones (top-down view of classroom):

          ┌───────────────────────┐
          │     TEACHER/SCREEN    │  ← yaw=0°, pitch=0°
          └───────────┬───────────┘
                      │
           ╔══════════╧══════════╗
           ║   ATTENTION ZONE    ║  ← ±20° yaw, ±15° pitch
           ║    (green zone)     ║
           ╠═════════════════════╣
           ║   MARGINAL ZONE     ║  ← ±30° yaw, ±25° pitch
           ║    (yellow zone)    ║
           ╠═════════════════════╣
           ║   INATTENTIVE       ║  ← beyond marginal
           ║    (red zone)       ║
           ╚═════════════════════╝
                      │
                  [STUDENT]

Features:
  - Configurable teacher/screen position
  - Three-zone classification (attentive / marginal / inattentive)
  - Temporal smoothing (brief glances don't penalise)
  - Attention score (0-100%)
"""
from dataclasses import dataclass
from collections import deque
from .config import cfg
from .pose_calculator import PoseResult


@dataclass
class AttentionResult:
    """Student attention assessment result."""
    is_attentive: bool  # In the green zone
    zone: str  # "attentive", "marginal", "inattentive"
    attention_score: float  # 0.0 to 1.0
    yaw_deviation: float  # Degrees from teacher
    pitch_deviation: float
    sustained_inattention: int  # Consecutive inattentive frames
    description: str


class AttentionJudge:
    """
    Judges if a student is paying attention based on head direction
    relative to the teacher/screen position.

    Usage:
        judge = AttentionJudge()
        result = judge.evaluate(pose_result)
        print(result.zone, result.attention_score)
    """

    def __init__(self):
        self._inattentive_counter = 0
        self._history = deque(maxlen=60)  # ~2 seconds at 30fps
        self._score_history = deque(maxlen=90)

    def evaluate(self, pose: PoseResult) -> AttentionResult:
        """
        Evaluate attention from head pose.

        Args:
            pose: PoseResult from PoseCalculator.

        Returns:
            AttentionResult with zone, score, and description.
        """
        # ── Deviation from teacher position ──
        yaw_dev = abs(pose.yaw - cfg.teacher_yaw)
        pitch_dev = abs(pose.pitch - cfg.teacher_pitch)

        # ── Zone classification ──
        yaw_tol = cfg.attention_yaw_tolerance
        pitch_tol = cfg.attention_pitch_tolerance

        if yaw_dev <= yaw_tol and pitch_dev <= pitch_tol:
            zone = "attentive"
            frame_score = 1.0
            self._inattentive_counter = 0
        elif yaw_dev <= yaw_tol * 1.5 and pitch_dev <= pitch_tol * 1.5:
            zone = "marginal"
            frame_score = 0.5
            self._inattentive_counter = 0
        else:
            zone = "inattentive"
            frame_score = 0.0
            self._inattentive_counter += 1

        # ── Smoothed attention score ──
        self._score_history.append(frame_score)
        attention_score = sum(self._score_history) / len(self._score_history)

        # ── Description ──
        if zone == "attentive":
            desc = "Looking at the teacher/screen ✓"
        elif zone == "marginal":
            desc = "Partially looking away — borderline attention"
        else:
            if self._inattentive_counter > 30:
                desc = f"⚠ Not looking at screen for {self._inattentive_counter / 30:.1f}s"
            else:
                desc = "Looking away from teacher/screen"

        return AttentionResult(
            is_attentive=(zone == "attentive"),
            zone=zone,
            attention_score=round(attention_score, 2),
            yaw_deviation=round(yaw_dev, 1),
            pitch_deviation=round(pitch_dev, 1),
            sustained_inattention=self._inattentive_counter,
            description=desc,
        )

    def reset(self):
        self._inattentive_counter = 0
        self._history.clear()
        self._score_history.clear()

