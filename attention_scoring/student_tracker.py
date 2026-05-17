

"""
attention_scoring/student_tracker.py
Manages multiple students simultaneously with per-student
scoring state, history, and statistics.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque
from .config import cfg
from .signal_fusion import SignalFusion, ComponentScores
from .scorer import AttentionScorer
from .state_classifier import StateClassifier, AttentionState, ClassificationResult


@dataclass
class StudentRecord:
    """Complete attention record for one student."""
    student_id: int
    current_score: float = 1.0
    current_state: AttentionState = AttentionState.ATTENTIVE
    classification: Optional[ClassificationResult] = None
    components: Optional[ComponentScores] = None

    # Statistics
    total_frames: int = 0
    attentive_frames: int = 0
    distracted_frames: int = 0
    sleepy_frames: int = 0
    looking_away_frames: int = 0

    # Time tracking
    last_seen: float = 0.0
    session_start: float = 0.0

    # History (last 60 seconds of scores)
    score_history: deque = field(default_factory=lambda: deque(maxlen=1800))

    @property
    def attention_percentage(self) -> float:
        """Overall attention % for this student."""
        if self.total_frames == 0:
            return 100.0
        return round(
            (self.attentive_frames / self.total_frames) * 100, 1
        )

    @property
    def state_distribution(self) -> Dict[str, float]:
        """Percentage breakdown of time in each state."""
        total = max(self.total_frames, 1)
        return {
            "attentive": round(self.attentive_frames / total * 100, 1),
            "distracted": round(self.distracted_frames / total * 100, 1),
            "sleepy": round(self.sleepy_frames / total * 100, 1),
            "looking_away": round(self.looking_away_frames / total * 100, 1),
        }

    @property
    def recent_score_avg(self) -> float:
        """Average score over last 5 seconds (~150 frames)."""
        recent = list(self.score_history)[-150:]
        if not recent:
            return 1.0
        return round(sum(recent) / len(recent), 3)

    def to_dict(self) -> dict:
        """Serialisable snapshot."""
        return {
            "student_id": self.student_id,
            "score": self.current_score,
            "state": self.current_state.value,
            "attention_pct": self.attention_percentage,
            "total_frames": self.total_frames,
            "distribution": self.state_distribution,
            "recent_avg": self.recent_score_avg,
            "trend": "stable",
        }


class StudentTracker:
    """
    Manages per-student scoring components.

    Each student gets their own:
      - SignalFusion instance
      - AttentionScorer (with own EMA state)
      - StateClassifier (with own hysteresis state)

    Usage:
        tracker = StudentTracker()
        record = tracker.update(student_id=0, yaw=5, pitch=3, ...)
        record = tracker.update(student_id=1, yaw=-30, pitch=0, ...)

        # Get all students
        for sid, rec in tracker.students.items():
            print(f"Student {sid}: {rec.attention_percentage:.0f}%")
    """

    def __init__(self):
        self._students: Dict[int, StudentRecord] = {}
        self._fusions: Dict[int, SignalFusion] = {}
        self._scorers: Dict[int, AttentionScorer] = {}
        self._classifiers: Dict[int, StateClassifier] = {}

    def _ensure_student(self, student_id: int):
        """Lazy-init per-student components."""
        if student_id not in self._students:
            self._students[student_id] = StudentRecord(
                student_id=student_id,
                session_start=time.time(),
                last_seen=time.time(),
            )
            self._fusions[student_id] = SignalFusion()
            self._scorers[student_id] = AttentionScorer()
            self._classifiers[student_id] = StateClassifier()

    def update(self,
               student_id: int,
               yaw: float = 0.0,
               pitch: float = 0.0,
               gaze_direction: str = "center",
               ear: float = 0.30,
               blink_rate: float = 15.0,
               perclos: float = 0.05,
               drowsiness_level: str = "alert",
               head_direction: str = "forward",
               ) -> StudentRecord:
        """
        Update a student's attention score with new signals.

        Args:
            student_id:       Unique student identifier.
            yaw, pitch:       Head pose angles.
            gaze_direction:   From GazeEstimator.
            ear:              Eye Aspect Ratio.
            blink_rate:       Blinks/min from BlinkDetector.
            perclos:          % eye closure from BlinkDetector.
            drowsiness_level: From DrowsinessDetector.
            head_direction:   From DirectionClassifier.

        Returns:
            Updated StudentRecord.
        """
        self._ensure_student(student_id)

        fusion = self._fusions[student_id]
        scorer = self._scorers[student_id]
        classifier = self._classifiers[student_id]
        record = self._students[student_id]

        # 1. Compute component scores
        components = fusion.compute(
            yaw=yaw, pitch=pitch,
            gaze_direction=gaze_direction,
            ear=ear, blink_rate=blink_rate,
            perclos=perclos,
        )

        # 2. Smooth the score
        smoothed = scorer.update(components.raw_weighted_score)

        # 3. Classify state
        classification = classifier.classify(
            score=smoothed,
            components=components,
            drowsiness_level=drowsiness_level,
            head_direction=head_direction,
        )

        # 4. Update record
        record.current_score = smoothed
        record.current_state = classification.state
        record.classification = classification
        record.components = components
        record.total_frames += 1
        record.last_seen = time.time()
        record.score_history.append(smoothed)

        # Track state counts
        state_map = {
            AttentionState.ATTENTIVE: "attentive_frames",
            AttentionState.DISTRACTED: "distracted_frames",
            AttentionState.SLEEPY: "sleepy_frames",
            AttentionState.LOOKING_AWAY: "looking_away_frames",
        }
        attr = state_map.get(classification.state)
        if attr:
            setattr(record, attr, getattr(record, attr) + 1)

        return record

    @property
    def students(self) -> Dict[int, StudentRecord]:
        return self._students

    def get_student(self, student_id: int) -> Optional[StudentRecord]:
        return self._students.get(student_id)

    def get_class_average(self) -> float:
        """Average attention score across all active students."""
        if not self._students:
            return 1.0
        scores = [r.current_score for r in self._students.values()]
        return round(sum(scores) / len(scores), 3)

    def get_class_summary(self) -> Dict:
        """Class-level summary statistics."""
        total = len(self._students)
        if total == 0:
            return {"total": 0, "attentive": 0, "distracted": 0,
                    "sleepy": 0, "looking_away": 0, "avg_score": 1.0}

        counts = {s.value: 0 for s in AttentionState}
        for r in self._students.values():
            counts[r.current_state.value] += 1

        return {
            "total": total,
            "attentive": counts["attentive"],
            "distracted": counts["distracted"],
            "sleepy": counts["sleepy"],
            "looking_away": counts["looking_away"],
            "avg_score": self.get_class_average(),
            "avg_attention_pct": round(
                sum(r.attention_percentage for r in self._students.values()) / total, 1
            ),
        }

    def cleanup_stale(self):
        """Remove students not seen for timeout period."""
        now = time.time()
        stale = [
            sid for sid, r in self._students.items()
            if (now - r.last_seen) > cfg.student_timeout_sec
        ]
        for sid in stale:
            del self._students[sid]
            del self._fusions[sid]
            del self._scorers[sid]
            del self._classifiers[sid]

    def reset(self):
        """Clear all students."""
        self._students.clear()
        self._fusions.clear()
        self._scorers.clear()
        self._classifiers.clear()