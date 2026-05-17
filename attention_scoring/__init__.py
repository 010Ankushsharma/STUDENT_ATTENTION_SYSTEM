"""
AI Attention Scoring Module
============================
Fuses eye tracking, blink frequency, gaze direction, and head pose
signals into a unified attention score per student.

Attention States:
    - ATTENTIVE    : Engaged with the lecture
    - DISTRACTED   : Partially disengaged
    - SLEEPY       : Drowsy / falling asleep
    - LOOKING_AWAY : Not facing the screen/teacher

Core Components:
    - SignalFusion         : Combines multiple input signals
    - AttentionScorer      : Calculates per-student attention %
    - StateClassifier      : Maps score → attention state label
    - AlertEngine          : Triggers alerts on low attention
    - StudentTracker       : Multi-student simultaneous tracking
    - ScoringPipeline      : End-to-end orchestrator

Usage:
    from attention_scoring import ScoringPipeline
    pipeline = ScoringPipeline()
    result = pipeline.update(student_id=0, ear=0.28, gaze="center",
                             yaw=5.0, pitch=3.0, blink_rate=16.0,
                             perclos=0.04, drowsiness_level="alert")
    print(result.state, result.score)
"""
from .signal_fusion import SignalFusion
from .scorer import AttentionScorer
from .state_classifier import StateClassifier, AttentionState
from .alert_engine import AlertEngine, Alert
from .student_tracker import StudentTracker, StudentRecord
from .pipeline import ScoringPipeline, ScoringResult

__all__ = [
    "SignalFusion",
    "AttentionScorer",
    "StateClassifier",
    "AttentionState",
    "AlertEngine",
    "Alert",
    "StudentTracker",
    "StudentRecord",
    "ScoringPipeline",
    "ScoringResult",
]
__version__ = "1.0.0"