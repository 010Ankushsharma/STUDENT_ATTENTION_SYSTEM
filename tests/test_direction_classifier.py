
"""
tests/test_direction_classifier.py
Unit tests for direction classification from angles.
"""
import pytest
import numpy as np
from head_pose.direction_classifier import DirectionClassifier, DirectionResult
from head_pose.pose_calculator import PoseResult


def _make_pose(yaw=0.0, pitch=0.0, roll=0.0):
    """Create a minimal PoseResult with given angles."""
    return PoseResult(
        yaw=yaw, pitch=pitch, roll=roll,
        rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)),
        rmat=np.eye(3), nose_tip_2d=(320, 240),
        nose_end_2d=(320, 200),
    )


class TestDirectionClassifier:
    def setup_method(self):
        self.clf = DirectionClassifier()

    def test_forward(self):
        r = self.clf.classify(_make_pose(yaw=0, pitch=0))
        assert r.is_forward is True
        assert r.combined_label == "forward"
        assert r.severity == "none"

    def test_slight_left(self):
        r = self.clf.classify(_make_pose(yaw=-15))
        assert r.horizontal == "slight_left"
        assert r.is_forward is False

    def test_strong_left(self):
        r = self.clf.classify(_make_pose(yaw=-40))
        assert r.horizontal == "left"
        assert r.primary_direction == "left"
        assert r.severity == "strong"

    def test_slight_right(self):
        r = self.clf.classify(_make_pose(yaw=15))
        assert r.horizontal == "slight_right"

    def test_strong_right(self):
        r = self.clf.classify(_make_pose(yaw=40))
        assert r.horizontal == "right"

    def test_down(self):
        r = self.clf.classify(_make_pose(pitch=-30))
        assert r.vertical == "down"
        assert r.primary_direction == "down"

    def test_up(self):
        r = self.clf.classify(_make_pose(pitch=30))
        assert r.vertical == "up"

    def test_diagonal_down_left(self):
        r = self.clf.classify(_make_pose(yaw=-35, pitch=-30))
        assert "down" in r.combined_label and "left" in r.combined_label

    def test_tilted(self):
        r = self.clf.classify(_make_pose(roll=30))
        assert r.is_tilted is True

    def test_not_tilted(self):
        r = self.clf.classify(_make_pose(roll=10))
        assert r.is_tilted is False

    def test_forward_within_tolerance(self):
        """Angles within forward range should be forward."""
        r = self.clf.classify(_make_pose(yaw=8, pitch=-5))
        assert r.is_forward is True


class TestAttentionJudgeIntegration:
    def test_attentive_when_forward(self):
        from head_pose.attention_judge import AttentionJudge
        judge = AttentionJudge()
        pose = _make_pose(yaw=5, pitch=3)
        result = judge.evaluate(pose)
        assert result.zone == "attentive"
        assert result.is_attentive is True

    def test_inattentive_when_looking_away(self):
        from head_pose.attention_judge import AttentionJudge
        judge = AttentionJudge()
        pose = _make_pose(yaw=45, pitch=0)
        result = judge.evaluate(pose)
        assert result.zone == "inattentive"
        assert result.is_attentive is False

    def test_marginal_zone(self):
        from head_pose.attention_judge import AttentionJudge
        judge = AttentionJudge()
        pose = _make_pose(yaw=25, pitch=0)
        result = judge.evaluate(pose)
        assert result.zone == "marginal"

    def test_sustained_inattention_counter(self):
        from head_pose.attention_judge import AttentionJudge
        judge = AttentionJudge()
        pose = _make_pose(yaw=50, pitch=0)
        for _ in range(10):
            result = judge.evaluate(pose)
        assert result.sustained_inattention == 10

    def test_attention_score_decay(self):
        from head_pose.attention_judge import AttentionJudge
        judge = AttentionJudge()
        # Start attentive
        for _ in range(20):
            judge.evaluate(_make_pose(yaw=0, pitch=0))
        # Go inattentive
        for _ in range(20):
            result = judge.evaluate(_make_pose(yaw=50))
        # Score should have decreased
        assert result.attention_score < 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
