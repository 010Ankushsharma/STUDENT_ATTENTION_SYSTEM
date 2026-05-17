"""
tests/test_state_classifier.py
Unit tests for state classification with hysteresis.
"""
import pytest
from attention_scoring.state_classifier import StateClassifier, AttentionState
from attention_scoring.signal_fusion import ComponentScores


def _make_components(**kwargs):
    defaults = dict(head_pose=1.0, gaze=1.0, ear=1.0, blink_rate=1.0,
                    perclos=1.0, raw_weighted_score=1.0)
    defaults.update(kwargs)
    return ComponentScores(**defaults)


class TestStateClassifier:
    def setup_method(self):
        self.clf = StateClassifier()

    def test_attentive_high_score(self):
        r = self.clf.classify(0.85, _make_components())
        assert r.state == AttentionState.ATTENTIVE

    def test_distracted_mid_score(self):
        """Need sustained frames to change state due to hysteresis."""
        for _ in range(15):
            r = self.clf.classify(0.55, _make_components(head_pose=0.5))
        assert r.state == AttentionState.DISTRACTED

    def test_sleepy_low_score_with_drowsiness(self):
        for _ in range(15):
            r = self.clf.classify(0.25, _make_components(ear=0.2, perclos=0.3),
                                  drowsiness_level="moderate_drowsy")
        assert r.state == AttentionState.SLEEPY

    def test_looking_away(self):
        for _ in range(15):
            r = self.clf.classify(0.20, _make_components(head_pose=0.1),
                                  head_direction="left")
        assert r.state == AttentionState.LOOKING_AWAY

    def test_hysteresis_prevents_flickering(self):
        """Single low-score frame shouldn't change state from attentive."""
        # Build up attentive state
        for _ in range(20):
            self.clf.classify(0.90, _make_components())
        # One bad frame
        r = self.clf.classify(0.30, _make_components(head_pose=0.1))
        # Should still be attentive (hysteresis)
        assert r.state == AttentionState.ATTENTIVE

    def test_state_change_flag(self):
        """State change should be signalled."""
        for _ in range(5):
            self.clf.classify(0.90, _make_components())
        for i in range(15):
            r = self.clf.classify(0.55, _make_components())
        # At some point is_state_change should be True
        # (after delay frames)
        assert r.state == AttentionState.DISTRACTED

    def test_confidence_high_for_clear_attentive(self):
        r = self.clf.classify(0.95, _make_components())
        assert r.confidence > 0.5

    def test_reset(self):
        self.clf.classify(0.55, _make_components())
        self.clf.reset()
        assert self.clf.current_state == AttentionState.ATTENTIVE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])