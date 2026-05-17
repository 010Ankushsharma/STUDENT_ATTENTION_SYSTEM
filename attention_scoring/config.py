"""
attention_scoring/config.py
All configurable parameters for attention scoring.
Modify these to tune sensitivity for different classroom scenarios.
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ScoringConfig:
    """
    Master configuration for the attention scoring system.

    Signal Weights:
        Each input signal contributes a weighted portion to the
        final attention score. Weights must sum to 1.0.

    Thresholds:
        Define boundaries between attention states.
    """

    # ─────────── Signal Weights ───────────
    #
    # How much each signal contributes to the final score.
    # Total must equal 1.0.
    #
    #   head_pose   : Most reliable indicator (where are they looking?)
    #   gaze        : Fine-grained attention direction
    #   ear         : Eye openness (drowsiness)
    #   blink_rate  : Fatigue indicator
    #   perclos     : Standard drowsiness metric
    #
    weights: Dict[str, float] = field(default_factory=lambda: {
        "head_pose":  0.35,     # Yaw + pitch contribution
        "gaze":       0.20,     # Iris-based gaze direction
        "ear":        0.20,     # Eye Aspect Ratio (openness)
        "blink_rate": 0.10,     # Blinks per minute
        "perclos":    0.15,     # % eye closure
    })

    # ─────────── Head Pose Scoring ───────────
    #
    # Score = 1.0 when facing forward, decreases with deviation.
    # Uses a smooth sigmoid-like decay.
    #
    head_yaw_full_score: float = 10.0       # Below this yaw → full score
    head_yaw_zero_score: float = 45.0       # Above this yaw → zero score
    head_pitch_full_score: float = 8.0
    head_pitch_zero_score: float = 35.0

    # ─────────── Gaze Scoring ───────────
    #
    # center = 1.0, slight deviation = 0.6, strong = 0.1
    #
    gaze_scores: Dict[str, float] = field(default_factory=lambda: {
        "center":       1.0,
        "up":           0.5,    # Could be reading top of board
        "down":         0.4,    # Could be taking notes
        "left":         0.2,
        "right":        0.2,
        "up-left":      0.3,
        "up-right":     0.3,
        "down-left":    0.2,
        "down-right":   0.2,
    })

    # ─────────── EAR (Eye Aspect Ratio) Scoring ───────────
    #
    # Open eyes = high EAR = high score
    #
    ear_full_score: float = 0.26        # Above this → 1.0
    ear_zero_score: float = 0.15        # Below this → 0.0
    ear_closed_score: float = 0.0       # Completely closed

    # ─────────── Blink Rate Scoring ───────────
    #
    # Normal blink rate: 12-20 per minute
    # Too high (fatigue) or too low (staring/zoned out) penalised.
    #
    blink_rate_optimal_low: float = 10.0
    blink_rate_optimal_high: float = 22.0
    blink_rate_penalty_high: float = 35.0   # Above this → score drops fast
    blink_rate_penalty_low: float = 4.0     # Below this → score drops

    # ─────────── PERCLOS Scoring ───────────
    #
    # PERCLOS = % time eyes closed in sliding window
    # 0% = perfect, >15% = drowsy
    #
    perclos_full_score: float = 0.05    # Below 5% → 1.0
    perclos_zero_score: float = 0.25    # Above 25% → 0.0

    # ─────────── State Classification Thresholds ───────────
    #
    #   Score Range          State
    #   ──────────────       ──────────────
    #   0.75 - 1.00         ATTENTIVE
    #   0.45 - 0.74         DISTRACTED
    #   0.00 - 0.44         SLEEPY / LOOKING_AWAY
    #
    threshold_attentive: float = 0.75
    threshold_distracted: float = 0.45
    # Below distracted → sleepy or looking_away (determined by signals)

    # ─────────── Temporal Smoothing ───────────
    #
    # Prevents rapid flickering between states.
    # Score is smoothed over N frames.
    #
    smoothing_window: int = 15          # ~0.5 sec at 30fps
    state_change_delay: int = 8         # Frames before confirming state change
    score_momentum: float = 0.3         # Exponential moving average factor
                                        # Higher = faster response

    # ─────────── Alert Configuration ───────────
    #
    alert_cooldown_sec: float = 30.0        # Min seconds between alerts per student
    alert_score_threshold: float = 0.40     # Score below this triggers alert
    alert_sustained_frames: int = 45        # ~1.5 sec before alerting
    alert_sleepy_immediate: bool = True     # Sleepy state always alerts immediately
    alert_max_per_session: int = 50         # Cap alerts per student per session

    # ─────────── Multi-Student ───────────
    #
    max_students: int = 30
    student_timeout_sec: float = 10.0       # Remove student after N sec no detection


cfg = ScoringConfig()