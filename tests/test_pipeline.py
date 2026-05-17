
"""
tests/test_pipeline.py
Integration tests for the full scoring pipeline.
"""
import pytest
from attention_scoring import ScoringPipeline, Alert
from attention_scoring.state_classifier import AttentionState


class TestScoringPipeline:
    def setup_method(self):
        self.pipe = ScoringPipeline()

    def test_attentive_student(self):
        for _ in range(30):
            r = self.pipe.update(
                student_id=0, ear=0.29, gaze_direction="center",
                yaw=3.0, pitch=-2.0, blink_rate=16.0, perclos=0.03,
            )
        assert r.score > 0.7
        assert r.state == "attentive"
        assert r.is_attentive is True

    def test_sleepy_student(self):
        for _ in range(50):
            r = self.pipe.update(
                student_id=1, ear=0.14, gaze_direction="down",
                yaw=-3.0, pitch=-12.0, blink_rate=28.0, perclos=0.22,
                drowsiness_level="moderate_drowsy",
            )
        assert r.state == "sleepy"
        assert r.score < 0.5

    def test_looking_away_student(self):
        for _ in range(50):
            r = self.pipe.update(
                student_id=2, ear=0.30, gaze_direction="left",
                yaw=-42.0, pitch=0.0, blink_rate=14.0, perclos=0.04,
                head_direction="left",
            )
        assert r.state == "looking_away"

    def test_multiple_students(self):
        """Pipeline handles multiple students simultaneously."""
        for _ in range(30):
            self.pipe.update(student_id=0, ear=0.30, yaw=0, pitch=0)
            self.pipe.update(student_id=1, ear=0.14, yaw=-40, pitch=0,
                            drowsiness_level="severe_drowsy",
                            head_direction="left")

        s0 = self.pipe.get_student(0)
        s1 = self.pipe.get_student(1)
        assert s0.current_score > s1.current_score
        assert s0.attention_percentage > s1.attention_percentage

    def test_class_summary(self):
        for _ in range(10):
            self.pipe.update(student_id=0, ear=0.30, yaw=0, pitch=0)
            self.pipe.update(student_id=1, ear=0.30, yaw=0, pitch=0)
        summary = self.pipe.get_class_summary()
        assert summary["total"] == 2

    def test_alert_triggered(self):
        alerts_received = []
        self.pipe.set_alert_callback(lambda a: alerts_received.append(a))

        for _ in range(60):
            self.pipe.update(
                student_id=5, ear=0.12, yaw=-5, pitch=-10,
                perclos=0.25, blink_rate=30,
                drowsiness_level="severe_drowsy",
                head_direction="down",
            )
        assert len(alerts_received) > 0
        assert alerts_received[0].student_id == 5

    def test_score_percentage(self):
        r = self.pipe.update(student_id=0, ear=0.30, yaw=0, pitch=0)
        assert 0 <= r.score_pct <= 100

    def test_components_returned(self):
        r = self.pipe.update(student_id=0, ear=0.30, yaw=0, pitch=0)
        assert "head_pose" in r.components
        assert "ear" in r.components

    def test_leaderboard(self):
        for _ in range(10):
            self.pipe.update(student_id=0, ear=0.30, yaw=0, pitch=0)
            self.pipe.update(student_id=1, ear=0.15, yaw=-40, pitch=0)
        lb = self.pipe.get_leaderboard()
        assert lb[0]["student_id"] == 0  # Best student first

    def test_reset(self):
        self.pipe.update(student_id=0, ear=0.30, yaw=0, pitch=0)
        self.pipe.reset()
        assert len(self.pipe.get_all_students()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
