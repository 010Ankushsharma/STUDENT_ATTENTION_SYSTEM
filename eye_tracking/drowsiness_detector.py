
"""
eye_tracking/drowsiness_detector.py
Multi-signal drowsiness / sleepiness classification.

Uses 4 indicators:
  1. Sustained low EAR (eyes drooping)
  2. High PERCLOS (eyes closed too often)
  3. Abnormal blink rate (too high = fatigue, too low = zoned out)
  4. Prolonged eye closure (> N frames)

Drowsiness Levels:
    ┌─────────────┬────────────────────────────────┐
    │ Level       │ Criteria                        │
    ├─────────────┼────────────────────────────────┤
    │ ALERT       │ All metrics normal               │
    │ MILD        │ 1 indicator triggered             │
    │ MODERATE    │ 2 indicators triggered            │
    │ SEVERE      │ 3+ indicators or prolonged close  │
    └─────────────┴────────────────────────────────┘
"""
from dataclasses import dataclass
from enum import Enum
from typing import List
from .config import cfg
from .blink_detector import BlinkMetrics


class DrowsinessLevel(str, Enum):
    ALERT = "alert"
    MILD = "mild_drowsy"
    MODERATE = "moderate_drowsy"
    SEVERE = "severe_drowsy"


@dataclass
class DrowsinessResult:
    """Drowsiness assessment for one student."""
    level: DrowsinessLevel
    score: float  # 0.0 (alert) to 1.0 (severe)
    triggers: List[str]  # Which indicators fired
    recommendation: str  # Human-readable suggestion
    ear: float
    perclos: float
    blink_rate: float
    is_eyes_closed: bool


class DrowsinessDetector:
    """
    Classifies drowsiness from blink metrics.

    Usage:
        detector = DrowsinessDetector()
        result = detector.assess(blink_metrics)
        print(result.level, result.score, result.triggers)
    """

    def __init__(self):
        self._low_ear_counter = 0  # Consecutive frames with low EAR

    def assess(self, metrics: BlinkMetrics) -> DrowsinessResult:
        """
        Assess drowsiness level from current blink metrics.

        Args:
            metrics: BlinkMetrics from BlinkDetector.update()

        Returns:
            DrowsinessResult with level, score, and triggers.
        """
        triggers: List[str] = []
        score = 0.0

        # ── Indicator 1: Sustained low EAR ──
        if metrics.current_ear < cfg.drowsy_ear_threshold:
            self._low_ear_counter += 1
        else:
            self._low_ear_counter = max(0, self._low_ear_counter - 2)  # Decay

        if self._low_ear_counter >= cfg.drowsy_consec_frames:
            triggers.append("sustained_low_ear")
            score += 0.35

        # ── Indicator 2: High PERCLOS ──
        if metrics.perclos > cfg.perclos_threshold:
            triggers.append(f"high_perclos ({metrics.perclos:.1%})")
            score += 0.30

        # ── Indicator 3: Abnormal blink rate ──
        if metrics.blink_rate_per_min > cfg.drowsy_blink_rate_high:
            triggers.append(f"rapid_blinking ({metrics.blink_rate_per_min:.0f}/min)")
            score += 0.15
        elif (metrics.blink_rate_per_min < cfg.drowsy_blink_rate_low
              and metrics.total_blinks > 5):  # Only after enough data
            triggers.append(f"low_blink_rate ({metrics.blink_rate_per_min:.0f}/min)")
            score += 0.10

        # ── Indicator 4: Prolonged eye closure ──
        if metrics.is_prolonged_closure:
            triggers.append(f"prolonged_closure ({metrics.closed_seconds:.1f}s)")
            score += 0.40

        # ── Classify level ──
        score = min(score, 1.0)
        num_triggers = len(triggers)

        if num_triggers == 0:
            level = DrowsinessLevel.ALERT
            recommendation = "Student is alert and attentive."
        elif num_triggers == 1:
            level = DrowsinessLevel.MILD
            recommendation = "Mild drowsiness detected — monitor closely."
        elif num_triggers == 2:
            level = DrowsinessLevel.MODERATE
            recommendation = "Moderate drowsiness — consider a break or engagement."
        else:
            level = DrowsinessLevel.SEVERE
            recommendation = "⚠️ Severe drowsiness — immediate attention needed!"

        # Override: prolonged closure is always severe
        if metrics.is_prolonged_closure and metrics.closed_seconds > 2.0:
            level = DrowsinessLevel.SEVERE
            score = max(score, 0.9)
            recommendation = "⚠️ Student appears to be sleeping!"

        return DrowsinessResult(
            level=level,
            score=round(score, 2),
            triggers=triggers,
            recommendation=recommendation,
            ear=metrics.current_ear,
            perclos=metrics.perclos,
            blink_rate=metrics.blink_rate_per_min,
            is_eyes_closed=metrics.eyes_closed,
        )

    def reset(self):
        self._low_ear_counter = 0

