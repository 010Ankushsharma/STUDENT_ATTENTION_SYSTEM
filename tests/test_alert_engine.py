"""
tests/test_alert_engine.py
Unit tests for the alert engine.
"""
import pytest
import time
from attention_scoring.alert_engine import AlertEngine, AlertType, AlertSeverity
from attention_scoring.state_classifier import AttentionState
from attention_scoring.config import cfg


class TestAlertEngine:
    def setup_method(self):
        self.engine = AlertEngine()
        # Override cooldown for testing
        cfg.alert_cooldown_sec = 0.1

    def test_no_alert_high_score(self):
        alerts = self.engine.check(0, 0.85, AttentionState.ATTENTIVE, 100)
        assert len(alerts) == 0

    def test_sleepy_immediate_alert(self):
        alerts = self.engine.check(0, 0.20, AttentionState.SLEEPY, 10)
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.SLEEPY
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_sustained_low_attention(self):
        alerts = self.engine.check(0, 0.30, AttentionState.DISTRACTED,
                                   sustained_frames=50)
        assert len(alerts) >= 1
        assert alerts[0].alert_type == AlertType.LOW_ATTENTION

    def test_cooldown_prevents_spam(self):
        self.engine.check(0, 0.20, AttentionState.SLEEPY, 10)
        # Immediate second check — should be blocked by cooldown
        alerts = self.engine.check(0, 0.20, AttentionState.SLEEPY, 15)
        assert len(alerts) == 0

    def test_cooldown_expires(self):
        cfg.alert_cooldown_sec = 0.05
        self.engine.check(0, 0.20, AttentionState.SLEEPY, 10)
        time.sleep(0.06)
        alerts = self.engine.check(0, 0.20, AttentionState.SLEEPY, 20)
        assert len(alerts) >= 1

    def test_callback_fires(self):
        received = []
        self.engine.set_callback(lambda a: received.append(a))
        self.engine.check(0, 0.20, AttentionState.SLEEPY, 10)
        assert len(received) == 1

    def test_different_students_independent(self):
        a1 = self.engine.check(0, 0.20, AttentionState.SLEEPY, 10)
        a2 = self.engine.check(1, 0.20, AttentionState.SLEEPY, 10)
        assert len(a1) >= 1
        assert len(a2) >= 1

    def test_reset(self):
        self.engine.check(0, 0.20, AttentionState.SLEEPY, 10)
        self.engine.reset()
        assert len(self.engine.all_alerts) == 0

    def teardown_method(self):
        cfg.alert_cooldown_sec = 30.0  # Restore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

