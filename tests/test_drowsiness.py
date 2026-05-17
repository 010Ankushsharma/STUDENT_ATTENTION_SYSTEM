
"""
tests/test_drowsiness.py
Unit tests for drowsiness classification.
"""
import pytest
from eye_tracking.blink_detector import BlinkMetrics
from eye_tracking.drowsiness_detector import DrowsinessDetector, DrowsinessLevel


def _make_metrics(**kwargs):
    defaults = dict(
        total_blinks=10,
        blink_rate_per_min=15.0,
        eyes_closed=False,
        closed_frames=0,
        closed_seconds=0.0,
        avg_blink_duration_ms=150.0,
        longest_closure_sec=0.3,
        perclos=0.05,
        current_ear=0.30,
        is_prolonged_closure=False,
    )
    defaults.update(kwargs)
    return BlinkMetrics(**defaults)


class TestDrowsinessDetector:
    def setup_method(self):
        self.det = DrowsinessDetector()

    def test_alert_state(self):
        m = _make_metrics()
        r = self.det.assess(m)
        assert r.level == DrowsinessLevel.ALERT

    def test_high_perclos_triggers(self):
        m = _make_metrics(perclos=0.25)
        r = self.det.assess(m)
        assert r.level != DrowsinessLevel.ALERT
        assert any("perclos" in t for t in r.triggers)

    def test_rapid_blinking_triggers(self):
        m = _make_metrics(blink_rate_per_min=30.0)
        r = self.det.assess(m)
        assert any("rapid" in t for t in r.triggers)

    def test_prolonged_closure_severe(self):
        m = _make_metrics(is_prolonged_closure=True, closed_seconds=3.0, eyes_closed=True)
        r = self.det.assess(m)
        assert r.level == DrowsinessLevel.SEVERE

    def test_sustained_low_ear(self):
        """Feed many low-EAR frames to trigger sustained indicator."""
        for i in range(20):
            m = _make_metrics(current_ear=0.15)
            r = self.det.assess(m)
        assert any("sustained" in t for t in r.triggers)

    def test_score_capped_at_1(self):
        m = _make_metrics(
            perclos=0.30, blink_rate_per_min=30.0,
            is_prolonged_closure=True, closed_seconds=5.0,
            current_ear=0.10, eyes_closed=True,
        )
        # Feed sustained low EAR
        for _ in range(20):
            r = self.det.assess(m)
        assert r.score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
