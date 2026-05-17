
"""
tests/test_core.py
Tests for core utilities (logger, errors, performance).
"""
import time
import pytest
from core.errors import (
    AttentionSystemError, CameraError, ProcessingError,
    DatabaseError, ConfigError, FaceDetectionError,
)
from core.performance import FPSCounter, Timer, PerformanceMonitor


class TestExceptions:
    def test_base_error(self):
        err = AttentionSystemError("Something failed", module="test")
        assert "test" in str(err)
        assert err.module == "test"

    def test_camera_error(self):
        err = CameraError("No camera", camera_index=2)
        assert err.camera_index == 2
        assert "camera" in str(err)

    def test_processing_error(self):
        err = ProcessingError("Frame failed", frame_num=42)
        assert err.frame_num == 42

    def test_face_detection_error(self):
        err = FaceDetectionError("No faces", frame_num=10)
        assert err.module == "face_detection"

    def test_database_error(self):
        err = DatabaseError("Connection lost", operation="INSERT")
        assert err.operation == "INSERT"

    def test_config_error(self):
        err = ConfigError("Invalid value", parameter="camera_index")
        assert err.parameter == "camera_index"

    def test_inheritance(self):
        err = CameraError("test")
        assert isinstance(err, AttentionSystemError)
        assert isinstance(err, Exception)


class TestFPSCounter:
    def test_initial_zero(self):
        fps = FPSCounter()
        assert fps.get() == 0.0

    def test_counts_fps(self):
        fps = FPSCounter(window=10)
        for _ in range(10):
            fps.tick()
            time.sleep(0.01)  # ~100 FPS
        result = fps.get()
        assert 50 < result < 200  # Reasonable range

    def test_ms_per_frame(self):
        fps = FPSCounter()
        for _ in range(5):
            fps.tick()
            time.sleep(0.02)
        ms = fps.get_ms_per_frame()
        assert ms > 0


class TestTimer:
    def test_basic_timing(self):
        with Timer("test") as t:
            time.sleep(0.01)
        assert t.elapsed_ms > 5  # At least 5ms
        assert t.elapsed_ms < 50  # Not too long
        assert t.name == "test"

    def test_zero_work(self):
        with Timer() as t:
            pass
        assert t.elapsed_ms >= 0
        assert t.elapsed_ms < 5


class TestPerformanceMonitor:
    def test_record_and_report(self):
        monitor = PerformanceMonitor()
        monitor.record("stage_a", 10.0)
        monitor.record("stage_a", 12.0)
        monitor.record("stage_b", 5.0)

        report = monitor.get_report()
        assert "stage_a" in report
        assert "stage_b" in report
        assert report["stage_a"]["avg_ms"] == 11.0
        assert report["stage_b"]["total_calls"] == 1

    def test_frame_budget(self):
        monitor = PerformanceMonitor()
        monitor.record("detect", 10.0)
        monitor.record("track", 5.0)
        budget = monitor.get_total_frame_budget(target_fps=30)
        assert budget["budget_ms"] == pytest.approx(33.33, rel=0.01)
        assert budget["used_ms"] == 15.0
        assert budget["fits_budget"] is True

    def test_reset(self):
        monitor = PerformanceMonitor()
        monitor.record("x", 10.0)
        monitor.reset()
        assert monitor.get_report() == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

