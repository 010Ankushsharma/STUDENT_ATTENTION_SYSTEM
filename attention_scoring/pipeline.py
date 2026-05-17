"""
attention_scoring/pipeline.py
End-to-end Scoring Pipeline — the single entry point for
integrating attention scoring into the main system.

Usage:
    from attention_scoring import ScoringPipeline, ScoringResult

    pipeline = ScoringPipeline()
    pipeline.set_alert_callback(my_handler)

    # Each frame, per student:
    result = pipeline.update(
        student_id=0,
        ear=0.28,
        gaze_direction="center",
        yaw=5.0,
        pitch=3.0,
        blink_rate=16.0,
        perclos=0.04,
        drowsiness_level="alert",
        head_direction="forward",
    )

    print(result.state)            # "attentive"
    print(result.score)            # 0.87
    print(result.attention_pct)    # 92.3
    print(result.alerts)           # []
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from .config import cfg
from .student_tracker import StudentTracker, StudentRecord
from .alert_engine import AlertEngine, Alert
from .state_classifier import AttentionState


@dataclass
class ScoringResult:
    """Complete scoring output for one student update."""
    student_id: int
    score: float                    # Smoothed attention score (0-1)
    score_pct: float                # As percentage (0-100)
    state: str                      # "attentive", "distracted", etc.
    state_enum: AttentionState
    confidence: float               # Classification confidence
    attention_pct: float            # Session-long attention %
    is_attentive: bool
    is_state_change: bool
    frames_in_state: int
    alerts: List[Alert]             # Alerts triggered this frame
    components: Dict[str, float]    # Per-signal scores
    trend: str                      # "improving", "declining", "stable"
    class_average: float            # Whole-class average score


class ScoringPipeline:
    """
    Main entry point for the attention scoring system.

    Handles:
        - Multi-student tracking
        - Signal fusion + smoothing
        - State classification with hysteresis
        - Alert generation with cooldown
        - Class-level statistics

    Integration example (inside your camera processing loop):

        scoring = ScoringPipeline()

        for student in detected_students:
            result = scoring.update(
                student_id=student.track_id,
                ear=student.ear,
                yaw=student.yaw,
                pitch=student.pitch,
                gaze_direction=student.gaze,
                blink_rate=student.blink_rate,
                perclos=student.perclos,
                drowsiness_level=student.drowsiness,
                head_direction=student.direction,
            )
            # Use result.score, result.state, result.alerts...
    """

    def __init__(self):
        self._tracker = StudentTracker()
        self._alert_engine = AlertEngine()

    def set_alert_callback(self, callback: Callable[[Alert], None]):
        """Register callback for real-time alerts."""
        self._alert_engine.set_callback(callback)

    def update(self,
               student_id: int,
               ear: float = 0.30,
               gaze_direction: str = "center",
               yaw: float = 0.0,
               pitch: float = 0.0,
               blink_rate: float = 15.0,
               perclos: float = 0.05,
               drowsiness_level: str = "alert",
               head_direction: str = "forward",
               ) -> ScoringResult:
        """
        Process one frame of data for a student.

        Args:
            student_id:       Unique tracked ID.
            ear:              Eye Aspect Ratio (0-0.4).
            gaze_direction:   "center","left","right","up","down", etc.
            yaw:              Head yaw degrees (0=forward).
            pitch:            Head pitch degrees (0=forward).
            blink_rate:       Blinks per minute.
            perclos:          Fraction of eye closure (0-1).
            drowsiness_level: "alert","mild_drowsy","moderate_drowsy","severe_drowsy".
            head_direction:   "forward","left","right","up","down","slight_left", etc.

        Returns:
            ScoringResult with score, state, alerts, and more.
        """
        # 1. Update student tracker (fusion + smoothing + classification)
        record = self._tracker.update(
            student_id=student_id,
            yaw=yaw, pitch=pitch,
            gaze_direction=gaze_direction,
            ear=ear, blink_rate=blink_rate,
            perclos=perclos,
            drowsiness_level=drowsiness_level,
            head_direction=head_direction,
        )

        # 2. Check for alerts
        alerts = self._alert_engine.check(
            student_id=student_id,
            score=record.current_score,
            state=record.current_state,
            sustained_frames=record.classification.frames_in_state if record.classification else 0,
        )

        # 3. Build result
        components = {}
        if record.components:
            components = {
                "head_pose": record.components.head_pose,
                "gaze": record.components.gaze,
                "ear": record.components.ear,
                "blink_rate": record.components.blink_rate,
                "perclos": record.components.perclos,
            }

        scorer = self._tracker._scorers.get(student_id)
        trend = scorer.trend if scorer else "stable"

        return ScoringResult(
            student_id=student_id,
            score=record.current_score,
            score_pct=round(record.current_score * 100, 1),
            state=record.current_state.value,
            state_enum=record.current_state,
            confidence=record.classification.confidence if record.classification else 1.0,
            attention_pct=record.attention_percentage,
            is_attentive=(record.current_state == AttentionState.ATTENTIVE),
            is_state_change=record.classification.is_state_change if record.classification else False,
            frames_in_state=record.classification.frames_in_state if record.classification else 0,
            alerts=alerts,
            components=components,
            trend=trend,
            class_average=self._tracker.get_class_average(),
        )

    # ──────────── Class-level API ────────────

    def get_student(self, student_id: int) -> Optional[StudentRecord]:
        """Get full record for a specific student."""
        return self._tracker.get_student(student_id)

    def get_all_students(self) -> Dict[int, StudentRecord]:
        """Get all tracked students."""
        return self._tracker.students

    def get_class_summary(self) -> Dict:
        """Get class-level statistics."""
        return self._tracker.get_class_summary()

    def get_leaderboard(self) -> List[Dict]:
        """Students ranked by attention percentage."""
        students = sorted(
            self._tracker.students.values(),
            key=lambda r: r.attention_percentage,
            reverse=True,
        )
        return [r.to_dict() for r in students]

    def get_alerts(self) -> List[Alert]:
        """All alerts from this session."""
        return self._alert_engine.all_alerts

    def cleanup(self):
        """Remove stale students not seen recently."""
        self._tracker.cleanup_stale()

    def reset(self):
        """Full reset."""
        self._tracker.reset()
        self._alert_engine.reset()
