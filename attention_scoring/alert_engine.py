"""
attention_scoring/alert_engine.py
Generates alerts when students show sustained low attention.

Alert Types:
    - LOW_ATTENTION   : Score below threshold for sustained period
    - SLEEPY          : Drowsiness detected
    - LOOKING_AWAY    : Not facing screen for extended time
    - RAPID_DECLINE   : Score dropping quickly

Features:
    - Per-student cooldown (don't spam alerts)
    - Severity levels (warning / critical)
    - Max alerts per session cap
    - Callback support for external notification
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict
from enum import Enum
from .config import cfg
from .state_classifier import AttentionState


class AlertSeverity(str, Enum):
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    LOW_ATTENTION = "low_attention"
    SLEEPY = "sleepy"
    LOOKING_AWAY = "looking_away"
    RAPID_DECLINE = "rapid_decline"


@dataclass
class Alert:
    """Single alert event."""
    student_id: int
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    score: float
    state: AttentionState
    timestamp: float            # Unix time
    sustained_frames: int       # How many frames at low attention


class AlertEngine:
    """
    Monitors attention scores and generates alerts.

    Usage:
        engine = AlertEngine()
        engine.set_callback(my_alert_handler)

        # Each frame, per student:
        alerts = engine.check(student_id=0, score=0.3,
                              state=AttentionState.SLEEPY,
                              sustained_frames=50)
    """

    def __init__(self):
        self._last_alert_time: Dict[int, float] = {}  # student_id → timestamp
        self._alert_counts: Dict[int, int] = {}       # student_id → count
        self._all_alerts: List[Alert] = []
        self._callbacks: List[Callable[[Alert], None]] = []
        self._prev_scores: Dict[int, float] = {}

    def set_callback(self, callback: Callable[[Alert], None]):
        """Register a callback function called when alerts fire."""
        self._callbacks.append(callback)

    def check(self,
              student_id: int,
              score: float,
              state: AttentionState,
              sustained_frames: int = 0,
              ) -> List[Alert]:
        """
        Check if an alert should be triggered.

        Args:
            student_id:       Tracked student ID.
            score:            Current smoothed attention score.
            state:            Current classified state.
            sustained_frames: How many frames at current state.

        Returns:
            List of new alerts generated (may be empty).
        """
        now = time.time()
        alerts: List[Alert] = []

        # Check cooldown
        last_time = self._last_alert_time.get(student_id, 0)
        if (now - last_time) < cfg.alert_cooldown_sec:
            self._prev_scores[student_id] = score
            return alerts

        # Check max alerts cap
        count = self._alert_counts.get(student_id, 0)
        if count >= cfg.alert_max_per_session:
            return alerts

        # ── Check conditions ──

        # 1. Sleepy (immediate if configured)
        if state == AttentionState.SLEEPY and cfg.alert_sleepy_immediate:
            alert = Alert(
                student_id=student_id,
                alert_type=AlertType.SLEEPY,
                severity=AlertSeverity.CRITICAL,
                message=f"Student {student_id} appears to be sleeping!",
                score=score,
                state=state,
                timestamp=now,
                sustained_frames=sustained_frames,
            )
            alerts.append(alert)

        # 2. Sustained low attention
        elif (score < cfg.alert_score_threshold
              and sustained_frames >= cfg.alert_sustained_frames):

            if state == AttentionState.LOOKING_AWAY:
                alert_type = AlertType.LOOKING_AWAY
                msg = f"Student {student_id} not looking at screen ({sustained_frames/30:.1f}s)"
                severity = AlertSeverity.WARNING
            else:
                alert_type = AlertType.LOW_ATTENTION
                msg = f"Student {student_id} attention low ({score:.0%}) for {sustained_frames/30:.1f}s"
                severity = (AlertSeverity.CRITICAL if score < 0.2
                            else AlertSeverity.WARNING)

            alerts.append(Alert(
                student_id=student_id,
                alert_type=alert_type,
                severity=severity,
                message=msg,
                score=score,
                state=state,
                timestamp=now,
                sustained_frames=sustained_frames,
            ))

        # 3. Rapid decline
        prev = self._prev_scores.get(student_id, 1.0)
        if prev - score > 0.4:  # Dropped 40% in one update cycle
            alerts.append(Alert(
                student_id=student_id,
                alert_type=AlertType.RAPID_DECLINE,
                severity=AlertSeverity.WARNING,
                message=f"Student {student_id} attention dropped rapidly ({prev:.0%}→{score:.0%})",
                score=score,
                state=state,
                timestamp=now,
                sustained_frames=sustained_frames,
            ))

        # Record and notify
        if alerts:
            self._last_alert_time[student_id] = now
            self._alert_counts[student_id] = count + len(alerts)
            self._all_alerts.extend(alerts)
            for cb in self._callbacks:
                for a in alerts:
                    cb(a)

        self._prev_scores[student_id] = score
        return alerts

    @property
    def all_alerts(self) -> List[Alert]:
        return self._all_alerts

    def get_student_alerts(self, student_id: int) -> List[Alert]:
        return [a for a in self._all_alerts if a.student_id == student_id]

    def reset(self):
        self._last_alert_time.clear()
        self._alert_counts.clear()
        self._all_alerts.clear()
        self._prev_scores.clear()