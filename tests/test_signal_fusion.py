"""
tests/test_signal_fusion.py
Unit tests for signal fusion and scoring functions.
"""
import pytest
from attention_scoring.signal_fusion import SignalFusion, ComponentScores
from attention_scoring.config import cfg


class TestSignalFusion:
    def setup_method(self):
        self.fusion = SignalFusion()

    def test_perfect_attention(self):
        """All signals optimal → score near 1.0."""
        c = self.fusion.compute(
            yaw=0, pitch=0, gaze_direction="center",
            ear=0.30, blink_rate=15.0, perclos=0.02,
        )
        assert c.raw_weighted_score > 0.9

    def test_fully_distracted(self):
        """All signals bad → score near 0.0."""
        c = self.fusion.compute(
            yaw=50, pitch=40, gaze_direction="left",
            ear=0.12, blink_rate=35.0, perclos=0.30,
        )
        assert c.raw_weighted_score < 0.2

    def test_head_pose_only_deviation(self):
        """Only head turned, other signals fine."""
        c = self.fusion.compute(
            yaw=35, pitch=0, gaze_direction="center",
            ear=0.30, blink_rate=15.0, perclos=0.03,
        )
        assert c.head_pose < 0.5
        assert c.gaze == 1.0
        assert c.ear == 1.0

    def test_ear_closed(self):
        """Very low EAR → low ear score."""
        c = self.fusion.compute(
            yaw=0, pitch=0, gaze_direction="center",
            ear=0.12, blink_rate=15.0, perclos=0.03,
        )
        assert c.ear < 0.1

    def test_ear_open(self):
        """Normal EAR → high ear score."""
        c = self.fusion.compute(
            yaw=0, pitch=0, gaze_direction="center",
            ear=0.30, blink_rate=15.0, perclos=0.03,
        )
        assert c.ear == 1.0

    def test_gaze_center(self):
        c = self.fusion.compute(gaze_direction="center")
        assert c.gaze == 1.0

    def test_gaze_away(self):
        c = self.fusion.compute(gaze_direction="left")
        assert c.gaze < 0.5

    def test_blink_rate_optimal(self):
        c = self.fusion.compute(blink_rate=16.0)
        assert c.blink_rate == 1.0

    def test_blink_rate_too_high(self):
        c = self.fusion.compute(blink_rate=36.0)
        assert c.blink_rate < 0.4

    def test_perclos_low(self):
        c = self.fusion.compute(perclos=0.02)
        assert c.perclos == 1.0

    def test_perclos_high(self):
        c = self.fusion.compute(perclos=0.30)
        assert c.perclos == 0.0

    def test_weights_sum_to_one(self):
        total = sum(cfg.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_score_bounded(self):
        """Score should always be 0-1."""
        c = self.fusion.compute(yaw=100, pitch=100, ear=0, perclos=1.0, blink_rate=50)
        assert 0.0 <= c.raw_weighted_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
