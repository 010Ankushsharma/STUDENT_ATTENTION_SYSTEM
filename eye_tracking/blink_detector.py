
"""
eye_tracking/blink_detector.py
Blink Detection — counts blinks, measures duration, computes blink rate.

State Machine:
    ┌────────────┐    EAR < threshold    ┌──────────────┐
    │  EYES OPEN │ ─────────────────────>│  EYES CLOSED │
    │            │<──────────────────────│              │
    └────────────┘    EAR >= threshold   └──────────────┘
                      (if closed_frames
                       >= min_frames)
                      → blink_count += 1

Metrics tracked:
    - Total blink count
    - Blink rate (blinks per minute)
    - Current blink duration (frames & seconds)
    - Average blink duration
    - Longest closure duration
    - PERCLOS (% of time eyes are closed)
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque
from .config import cfg


@dataclass
class BlinkEvent:
    """Record of a single blink."""
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_seconds: float
    min_ear: float  # Lowest EAR during the blink
    timestamp: float  # Unix time when blink ended


@dataclass
class BlinkMetrics:
    """Current blink/eye-closure metrics snapshot."""
    total_blinks: int = 0
    blink_rate_per_min: float = 0.0  # Blinks per minute
    eyes_closed: bool = False  # Right now
    closed_frames: int = 0  # Current closure duration
    closed_seconds: float = 0.0
    avg_blink_duration_ms: float = 0.0
    longest_closure_sec: float = 0.0
    perclos: float = 0.0  # % eyes closed in window
    current_ear: float = 0.3
    is_prolonged_closure: bool = False  # Closure beyond blink threshold


class BlinkDetector:
    """
    Detects blinks from a stream of EAR values.

    Usage:
        detector = BlinkDetector()

        # Each frame:
        metrics = detector.update(ear_value, frame_number)
        print(f"Blinks: {metrics.total_blinks}, Rate: {metrics.blink_rate_per_min:.1f}/min")
    """

    def __init__(self, fps: float = 30.0):
        self._fps = fps
        self._threshold = cfg.ear_blink_threshold
        self._min_frames = cfg.blink_consec_frames
        self._max_blink_frames = cfg.max_blink_duration_frames

        # State
        self._closed_counter = 0  # Consecutive frames below threshold
        self._min_ear_in_closure = 1.0  # Track lowest EAR in current closure
        self._closure_start_frame = 0
        self._total_blinks = 0
        self._start_time = time.time()

        # Blink history
        self._blink_history: List[BlinkEvent] = []
        self._recent_blinks: deque = deque()  # Timestamps for rate calc

        # PERCLOS tracking
        self._perclos_window: deque = deque(maxlen=cfg.perclos_window_frames)

        # Duration tracking
        self._total_closed_frames = 0
        self._longest_closure = 0.0

    # ──────────── Public API ────────────

    def update(self, ear: float, frame_number: int = 0) -> BlinkMetrics:
        """
        Process one frame's EAR value.

        Args:
            ear:          Average EAR for this frame.
            frame_number: Sequential frame counter.

        Returns:
            BlinkMetrics snapshot.
        """
        now = time.time()
        eyes_closed = ear < self._threshold

        # Track PERCLOS (1 = closed, 0 = open)
        self._perclos_window.append(1 if eyes_closed else 0)

        if eyes_closed:
            # ── Eyes just closed or still closed ──
            if self._closed_counter == 0:
                self._closure_start_frame = frame_number
            self._closed_counter += 1
            self._min_ear_in_closure = min(self._min_ear_in_closure, ear)
            self._total_closed_frames += 1
        else:
            # ── Eyes just opened ──
            if self._closed_counter >= self._min_frames:
                # Valid blink (or prolonged closure)
                duration_frames = self._closed_counter
                duration_sec = duration_frames / self._fps

                if duration_frames <= self._max_blink_frames:
                    # Normal blink
                    self._total_blinks += 1
                    self._recent_blinks.append(now)

                    event = BlinkEvent(
                        start_frame=self._closure_start_frame,
                        end_frame=frame_number,
                        duration_frames=duration_frames,
                        duration_seconds=round(duration_sec, 3),
                        min_ear=round(self._min_ear_in_closure, 4),
                        timestamp=now,
                    )
                    self._blink_history.append(event)

                # Track longest closure
                if duration_sec > self._longest_closure:
                    self._longest_closure = duration_sec

            # Reset closure state
            self._closed_counter = 0
            self._min_ear_in_closure = 1.0

        # ── Compute metrics ──
        blink_rate = self._compute_blink_rate(now)
        avg_duration = self._compute_avg_duration()
        perclos = self._compute_perclos()

        return BlinkMetrics(
            total_blinks=self._total_blinks,
            blink_rate_per_min=round(blink_rate, 1),
            eyes_closed=eyes_closed,
            closed_frames=self._closed_counter,
            closed_seconds=round(self._closed_counter / self._fps, 2),
            avg_blink_duration_ms=round(avg_duration, 1),
            longest_closure_sec=round(self._longest_closure, 2),
            perclos=round(perclos, 3),
            current_ear=round(ear, 4),
            is_prolonged_closure=(
                    self._closed_counter > self._max_blink_frames and eyes_closed
            ),
        )

    @property
    def blink_history(self) -> List[BlinkEvent]:
        """Full blink event history."""
        return self._blink_history

    def reset(self):
        """Reset all state."""
        self._closed_counter = 0
        self._total_blinks = 0
        self._blink_history.clear()
        self._recent_blinks.clear()
        self._perclos_window.clear()
        self._start_time = time.time()
        self._total_closed_frames = 0
        self._longest_closure = 0.0

    # ──────────── Internal ────────────

    def _compute_blink_rate(self, now: float) -> float:
        """Blinks per minute using 60-second sliding window."""
        window = 60.0
        while self._recent_blinks and (now - self._recent_blinks[0]) > window:
            self._recent_blinks.popleft()
        return len(self._recent_blinks)  # count in last 60s = blinks/min

    def _compute_avg_duration(self) -> float:
        """Average blink duration in milliseconds."""
        if not self._blink_history:
            return 0.0
        # Use last 20 blinks
        recent = self._blink_history[-20:]
        avg_sec = sum(b.duration_seconds for b in recent) / len(recent)
        return avg_sec * 1000.0  # ms

    def _compute_perclos(self) -> float:
        """
        PERCLOS — Percentage of Eye CLOSure.
        Fraction of time eyes are closed in the tracking window.
        Standard drowsiness indicator (Dinges, 1998).
        """
        if len(self._perclos_window) == 0:
            return 0.0
        return sum(self._perclos_window) / len(self._perclos_window)

