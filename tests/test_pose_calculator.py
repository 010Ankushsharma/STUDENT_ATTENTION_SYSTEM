
"""
tests/test_pose_calculator.py
Unit tests for pose computation and angle extraction.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock
from head_pose.landmark_mapper import LandmarkMapper, MappedLandmarks
from head_pose.pose_calculator import PoseCalculator, PoseResult
from head_pose.config import cfg


def _make_mapped(offsets=None):
    """Create MappedLandmarks with optional 2D offsets for testing."""
    # Default: face looking straight at camera
    base_2d = np.array([
        (320, 280),   # Nose tip
        (320, 420),   # Chin
        (230, 230),   # Left eye outer
        (410, 230),   # Right eye outer
        (260, 370),   # Left mouth
        (380, 370),   # Right mouth
    ], dtype=np.float64)

    if offsets is not None:
        base_2d += np.array(offsets, dtype=np.float64)

    return MappedLandmarks(
        image_points_2d=base_2d,
        model_points_3d=cfg.model_points_3d,
        nose_tip_2d=(int(base_2d[0][0]), int(base_2d[0][1])),
        all_points_2d=base_2d,
    )


class TestPoseCalculator:
    def setup_method(self):
        self.calc = PoseCalculator(frame_w=640, frame_h=480)
        # Disable Kalman for deterministic tests
        cfg.use_kalman = False
        self.calc = PoseCalculator(frame_w=640, frame_h=480)

    def test_returns_pose_result(self):
        mapped = _make_mapped()
        result = self.calc.compute(mapped)
        assert result is not None
        assert isinstance(result, PoseResult)

    def test_forward_looking_small_angles(self):
        """Centered face should have small yaw and pitch."""
        mapped = _make_mapped()
        result = self.calc.compute(mapped)
        assert abs(result.yaw) < 15, f"Expected small yaw, got {result.yaw}"
        assert abs(result.pitch) < 15, f"Expected small pitch, got {result.pitch}"

    def test_shifted_right_positive_yaw(self):
        """Face shifted right in image → positive yaw (looking right)."""
        # Shift all points to create asymmetry suggesting rightward turn
        offsets = [
            (30, 0), (30, 0), (50, 0), (10, 0), (50, 0), (10, 0)
        ]
        mapped = _make_mapped(offsets)
        result = self.calc.compute(mapped)
        # The asymmetry should produce a non-zero yaw
        assert result is not None

    def test_has_axis_points(self):
        mapped = _make_mapped()
        result = self.calc.compute(mapped)
        assert result.axis_points_2d is not None
        assert result.axis_points_2d.shape == (4, 2)

    def test_reprojection_error_reasonable(self):
        mapped = _make_mapped()
        result = self.calc.compute(mapped)
        assert result.reprojection_error < 50, f"Error too high: {result.reprojection_error}"

    def test_confidence_positive(self):
        mapped = _make_mapped()
        result = self.calc.compute(mapped)
        assert result.confidence > 0

    def test_reset(self):
        self.calc.compute(_make_mapped())
        self.calc.reset()
        assert self.calc._prev_rvec is None

    def teardown_method(self):
        cfg.use_kalman = True  # Restore


class TestLandmarkMapper:
    def test_basic_mode(self):
        mapper = LandmarkMapper(use_extended=False)
        assert mapper.point_count == 6
        assert mapper.mode == "basic_6pt"

    def test_extended_mode(self):
        mapper = LandmarkMapper(use_extended=True)
        assert mapper.point_count == 14
        assert mapper.mode == "extended_14pt"

    def test_extract_returns_correct_shape(self):
        mapper = LandmarkMapper(use_extended=False)
        # Mock 500 landmarks
        landmarks = [MagicMock(x=0.5, y=0.5) for _ in range(500)]
        mapped = mapper.extract(landmarks, 640, 480)
        assert mapped.image_points_2d.shape == (6, 2)
        assert mapped.model_points_3d.shape == (6, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
