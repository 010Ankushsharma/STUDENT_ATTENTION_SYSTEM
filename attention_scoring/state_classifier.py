"""
attention_scoring/state_classifier.py
Maps smoothed attention score → attention state label.

Classification Logic:

    ┌──────────────────────────────────────────────────────────────┐
    │  Score ≥ 0.75                              → ATTENTIVE       │
    │  Score 0.45 - 0.74                         → DISTRACTED      │
    │  Score < 0.45 AND drowsiness detected      → SLEEPY          │
    │  Score < 0.45 AND head turned away         → LOOKING_AWAY    │
    │  Score < 0.45 (ambiguous)                  → DISTRACTED      │
    └──────────────────────────────────────────────────────────────┘

State-change hysteresis:
    State only changes after N consecutive frames vote for the new state.
    This prevents rapid flickering (e.g., momentary glances away).
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from .config import cfg
from .signal_fusion import ComponentScores


class AttentionState(str, Enum):
    """Student attention state labels."""
    ATTENTIVE = "attentive"
    DISTRACTED = "distracted"
    SLEEPY = "sleepy"
    LOOKING_AWAY = "looking_away"


@dataclass
class ClassificationResult:
    """Output of state classification."""
    state: AttentionState
    confidence: float           # How sure we are (0-1)
    score: float                # The smoothed attention score
    previous_state: AttentionState
    frames_in_state: int        # How long in current state
    is_state_change: bool       # Did state just change?


class StateClassifier:
    """
    Classifies attention state from score + signal context.

    Includes hysteresis to prevent flickering.

    Usage:
        classifier = StateClassifier()
        result = classifier.classify(
            score=0.82,
            components=component_scores,
            drowsiness_level="alert",
            head_direction="forward",
        )
        print(result.state)  # AttentionState.ATTENTIVE
    """

    def __init__(self):
        self._current_state = AttentionState.ATTENTIVE
        self._candidate_state: Optional[AttentionState] = None
        self._candidate_frames = 0
        self._frames_in_current = 0
        self._delay = cfg.state_change_delay

    def classify(self,
                 score: float,
                 components: ComponentScores,
                 drowsiness_level: str = "alert",
                 head_direction: str = "forward",
                 ) -> ClassificationResult:
        """
        Classify the current attention state.

        Args:
            score:             Smoothed attention score (0-1)
            components:        Per-signal scores from SignalFusion
            drowsiness_level:  From DrowsinessDetector ("alert","mild_drowsy", etc.)
            head_direction:    From DirectionClassifier ("forward","left", etc.)

        Returns:
            ClassificationResult with state and metadata.
        """
        # ── Determine raw state from score + context ──
        raw_state = self._compute_raw_state(
            score, components, drowsiness_level, head_direction
        )

        # ── Apply hysteresis ──
        is_change = False
        if raw_state != self._current_state:
            if raw_state == self._candidate_state:
                self._candidate_frames += 1
            else:
                self._candidate_state = raw_state
                self._candidate_frames = 1

            # Confirm state change after delay
            if self._candidate_frames >= self._delay:
                previous = self._current_state
                self._current_state = raw_state
                self._candidate_state = None
                self._candidate_frames = 0
                self._frames_in_current = 0
                is_change = True
        else:
            self._candidate_state = None
            self._candidate_frames = 0

        self._frames_in_current += 1

        # ── Confidence ──
        confidence = self._compute_confidence(score, raw_state)

        return ClassificationResult(
            state=self._current_state,
            confidence=round(confidence, 2),
            score=round(score, 4),
            previous_state=self._current_state if not is_change else previous,
            frames_in_state=self._frames_in_current,
            is_state_change=is_change,
        )

    def _compute_raw_state(self, score, components, drowsiness, direction) -> AttentionState:
        """Determine state without hysteresis."""
        if score >= cfg.threshold_attentive:
            return AttentionState.ATTENTIVE

        if score >= cfg.threshold_distracted:
            return AttentionState.DISTRACTED

        # Below distracted threshold — differentiate sleepy vs looking_away
        # Priority: sleepy signals override direction signals
        is_drowsy = drowsiness in ("mild_drowsy", "moderate_drowsy", "severe_drowsy")
        is_eyes_issue = (components.ear < 0.5 or components.perclos < 0.5)
        is_turned = (components.head_pose < 0.3)
        is_direction_away = direction not in ("forward", "center", "slight_left",
                                              "slight_right", "slight_up", "slight_down")

        if is_drowsy or (is_eyes_issue and components.ear < 0.3):
            return AttentionState.SLEEPY
        elif is_turned or is_direction_away:
            return AttentionState.LOOKING_AWAY
        else:
            return AttentionState.DISTRACTED

    @staticmethod
    def _compute_confidence(score: float, state: AttentionState) -> float:
        """How confident we are in the classification (distance from boundary)."""
        if state == AttentionState.ATTENTIVE:
            # Distance above attentive threshold
            return min(1.0, (score - cfg.threshold_attentive) / 0.25 + 0.6)
        elif state == AttentionState.DISTRACTED:
            mid = (cfg.threshold_attentive + cfg.threshold_distracted) / 2
            dist = 1.0 - abs(score - mid) / (
                (cfg.threshold_attentive - cfg.threshold_distracted) / 2)
            return max(0.4, min(1.0, dist))
        else:
            # Sleepy or looking away — confidence from how low the score is
            return min(1.0, (cfg.threshold_distracted - score) / 0.3 + 0.5)

    @property
    def current_state(self) -> AttentionState:
        return self._current_state

    def reset(self):
        self._current_state = AttentionState.ATTENTIVE
        self._candidate_state = None
        self._candidate_frames = 0
        self._frames_in_current = 0


