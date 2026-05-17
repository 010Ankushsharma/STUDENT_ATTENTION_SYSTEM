from collections import deque
from .config import cfg

class AttentionScorer:
    def __init__(self):
        self._ema_score: float = 1.0
        self._window = deque(maxlen=cfg.smoothing_window)
        self._momentum = cfg.score_momentum
        self._frame_count = 0

    def update(self, raw_score: float) -> float:
        self._frame_count += 1
        raw_score = max(0.0, min(1.0, raw_score))
        if self._frame_count <= 3:
            self._ema_score = raw_score
        else:
            self._ema_score = (
                self._momentum * raw_score
                + (1.0 - self._momentum) * self._ema_score
            )
        self._window.append(self._ema_score)
        return round(self._ema_score, 4)

    @property
    def current_score(self) -> float:
        return round(self._ema_score, 4)

    @property
    def window_average(self) -> float:
        if not self._window:
            return 1.0
        return round(sum(self._window) / len(self._window), 4)

    @property
    def trend(self) -> str:
        if len(self._window) < 10:
            return "stable"
        first_half = list(self._window)[:len(self._window)//2]
        second_half = list(self._window)[len(self._window)//2:]
        diff = sum(second_half)/len(second_half) - sum(first_half)/len(first_half)
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        return "stable"

    def reset(self):
        self._ema_score = 1.0
        self._window.clear()
        self._frame_count = 0
