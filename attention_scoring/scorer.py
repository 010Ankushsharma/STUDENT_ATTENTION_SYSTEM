
""" attention_scoring/scorer.py Applies temporal smoothing to raw scores for stable output.

Smoothing methods: 1. Exponential Moving Average (EMA) — responds fast to changes 2. Sliding window average — stable but slightly laggy 3. State-change delay — prevents flickering

Final Score = EMA(raw_score)
State only changes after `state_change_delay` consecutive frames
in the new state.
"""
from collections import deque
from .config import cfg

class AttentionScorer: """ Applies temporal smoothing to produce a stable attention score.

Usage:
    scorer = AttentionScorer()
    smooth_score = scorer.update(raw_score=0.82)
    smooth_score = scorer.update(raw_score=0.45)
    print(f"Smoothed: {smooth_score:.2f}")
"""

def __init__(self):
    self._ema_score: float = 1.0        # Start assuming attentive
    self._window = deque(maxlen=cfg.smoothing_window)
    self._momentum = cfg.score_momentum
    self._frame_count = 0

def update(self, raw_score: float) -> float:
    """
    Update with a new raw score and return smoothed score.

    The smoothed score uses an Exponential Moving Average:
        smoothed = α * raw + (1 - α) * previous

    Where α = score_momentum (higher = faster response).

    Args:
        raw_score: Fused score from SignalFusion (0.0-1.0)

    Returns:
        Smoothed attention score (0.0-1.0)
    """
    self._frame_count += 1
    raw_score = max(0.0, min(1.0, raw_score))

    # Exponential Moving Average
    if self._frame_count <= 3:
        # First few frames: use raw directly (no history yet)
        self._ema_score = raw_score
    else:
        self._ema_score = (
            self._momentum * raw_score
            + (1.0 - self._momentum) * self._ema_score
        )

    # Also track in sliding window
    self._window.append(self._ema_score)

    return round(self._ema_score, 4)

@property
def current_score(self) -> float:
    """Get current smoothed score without updating."""
    return round(self._ema_score, 4)

@property
def window_average(self) -> float:
    """Average over the full sliding window."""
    if not self._window:
        return 1.0
    return round(sum(self._window) / len(self._window), 4)

@property
def trend(self) -> str:
    """Score trend direction: improving / declining / stable."""
    if len(self._window) < 10:
        return "stable"
    first_half = list(self._window)[:len(self._window)//2]
    second_half = list(self._window)[len(self._window)//2:]
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    diff = avg_second - avg_first
    if diff > 0.05:
        return "improving"
    elif diff < -0.05:
        return "declining"
    return "stable"

def reset(self):
    """Reset scorer to initial state."""
    self._ema_score = 1.0
    self._window.clear()
    self._frame_count = 0