
"""
tests/test_blink_detector.py
Unit tests for blink detection logic.
"""
import pytest
from eye_tracking.blink_detector import BlinkDetector


class TestBlinkDetector:
    def setup_method(self):
        self.det = BlinkDetector(fps=30.0)

    def test_no_blink_with_open_eyes(self):
        """Normal open EAR should produce no blinks."""
        for i in range(60):
            m = self.det.update(0.30, i)
        assert m.total_blinks == 0

    def test_single_blink(self):
        """Drop EAR below threshold for 3 frames, then back up = 1 blink."""
        frame = 0
        # Open
        for _ in range(10):
            self.det.update(0.30, frame); frame += 1
        # Closed (3 frames)
        for _ in range(3):
            self.det.update(0.12, frame); frame += 1
        # Re-open
        m = self.det.update(0.30, frame)
        assert m.total_blinks == 1

    def test_multiple_blinks(self):
        """3 blink sequences = 3 blinks."""
        frame = 0
        for _ in range(3):
            # Open
            for _ in range(10):
                self.det.update(0.30, frame); frame += 1
            # Closed
            for _ in range(3):
                self.det.update(0.12, frame); frame += 1
        # Final open to register last blink
        m = self.det.update(0.30, frame)
        assert m.total_blinks == 3

    def test_too_short_not_a_blink(self):
        """1 frame below threshold shouldn't count (needs >= 2)."""
        frame = 0
        for _ in range(10):
            self.det.update(0.30, frame); frame += 1
        self.det.update(0.12, frame); frame += 1
        m = self.det.update(0.30, frame)
        assert m.total_blinks == 0

    def test_prolonged_closure(self):
        """Long closure beyond max_blink_duration = not counted as blink."""
        frame = 0
        for _ in range(10):
            self.det.update(0.30, frame); frame += 1
        # Closed for 20 frames (beyond max_blink_duration_frames=8)
        for _ in range(20):
            m = self.det.update(0.12, frame); frame += 1
        assert m.is_prolonged_closure is True
        # Re-open — should NOT increment blink count
        m = self.det.update(0.30, frame)
        assert m.total_blinks == 0

    def test_perclos_tracking(self):
        """PERCLOS should reflect fraction of closed frames."""
        frame = 0
        # 5 open, 5 closed = 50% PERCLOS
        for _ in range(5):
            self.det.update(0.30, frame); frame += 1
        for _ in range(5):
            m = self.det.update(0.12, frame); frame += 1
        assert 0.4 <= m.perclos <= 0.6, f"Expected ~0.5, got {m.perclos}"

    def test_eyes_closed_flag(self):
        m = self.det.update(0.12, 0)
        assert m.eyes_closed is True
        m = self.det.update(0.30, 1)
        assert m.eyes_closed is False

    def test_reset(self):
        self.det.update(0.12, 0)
        self.det.update(0.12, 1)
        self.det.update(0.12, 2)
        self.det.update(0.30, 3)
        assert self.det._total_blinks >= 1
        self.det.reset()
        assert self.det._total_blinks == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

