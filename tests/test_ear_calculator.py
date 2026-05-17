
"""
tests/test_ear_calculator.py
Unit tests for EAR calculation.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock
from eye_tracking.ear_calculator import EARCalculator


def _make_landmarks(left_pts, right_pts):
    """Create mock landmarks with given eye points."""
    landmarks = [MagicMock() for _ in range(500)]
    # Left eye indices: [362, 385, 387, 263, 373, 380]
    left_idx = [362, 385, 387, 263, 373, 380]
    # Right eye indices: [33, 160, 158, 133, 153, 144]
    right_idx = [33, 160, 158, 133, 153, 144]
    w, h = 640, 480
    for i, (px, py) in zip(left_idx, left_pts):
        landmarks[i].x = px / w
        landmarks[i].y = py / h
    for i, (px, py) in zip(right_idx, right_pts):
        landmarks[i].x = px / w
        landmarks[i].y = py / h
    return landmarks


class TestEARCalculator:
    def setup_method(self):
        self.calc = EARCalculator()
        self.calc._history.clear()

    def test_open_eye_ear(self):
        """Open eyes should have EAR ≈ 0.3."""
        # Simulated open eye: wide horizontal, moderate vertical
        left = [(100, 200), (120, 180), (150, 178), (180, 200), (150, 220), (120, 222)]
        right = [(300, 200), (320, 180), (350, 178), (380, 200), (350, 220), (320, 222)]
        lm = _make_landmarks(left, right)
        l, r, avg = self.calc.compute(lm, 640, 480)
        assert 0.2 < avg < 0.5, f"Open eye EAR should be ~0.3, got {avg}"

    def test_closed_eye_ear(self):
        """Closed eyes should have EAR < 0.15."""
        # Simulated closed eye: vertical distances very small
        left = [(100, 200), (120, 198), (150, 197), (180, 200), (150, 202), (120, 203)]
        right = [(300, 200), (320, 198), (350, 197), (380, 200), (350, 202), (320, 203)]
        lm = _make_landmarks(left, right)
        l, r, avg = self.calc.compute(lm, 640, 480)
        assert avg < 0.15, f"Closed eye EAR should be < 0.15, got {avg}"

    def test_symmetry(self):
        """Same geometry for both eyes should give equal EAR."""
        pts = [(100, 200), (120, 180), (150, 178), (180, 200), (150, 220), (120, 222)]
        lm = _make_landmarks(pts, pts)
        l, r, avg = self.calc.compute(lm, 640, 480)
        assert abs(l - r) < 0.05, f"Symmetric eyes should have similar EAR: L={l}, R={r}"

    def test_smoothing(self):
        """Smoothed EAR should dampen sudden jumps."""
        pts_open = [(100, 200), (120, 180), (150, 178), (180, 200), (150, 220), (120, 222)]
        pts_closed = [(100, 200), (120, 198), (150, 197), (180, 200), (150, 202), (120, 203)]
        lm_open = _make_landmarks(pts_open, pts_open)
        lm_closed = _make_landmarks(pts_closed, pts_closed)

        # Feed several open frames
        for _ in range(4):
            self.calc.compute(lm_open, 640, 480)
        # Sudden close — smoothed value shouldn't drop instantly to closed level
        _, _, avg = self.calc.compute(lm_closed, 640, 480)
        assert avg > 0.10, "Smoothing should dampen sudden EAR drop"

    def test_reset(self):
        self.calc._history = [0.3, 0.3, 0.3]
        self.calc.reset()
        assert len(self.calc._history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
