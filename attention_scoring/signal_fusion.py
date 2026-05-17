
"""
attention_scoring/signal_fusion.py
Converts raw sensor signals into normalised component scores (0.0-1.0)
and fuses them into a single weighted attention score.

Signal Processing Pipeline:
    Raw Inputs → Normalise (0-1) → Weight → Sum → Final Score

Each signal has its own scoring function that maps the raw value
to a 0.0-1.0 range using smooth interpolation (not hard cutoffs).
"""
import math
from dataclasses import dataclass
from typing import Optional, Dict
from .config import cfg


@dataclass
class ComponentScores:
    """Individual normalised scores per signal (all 0.0 to 1.0)."""
    head_pose: float = 1.0
    gaze: float = 1.0
    ear: float = 1.0
    blink_rate: float = 1.0
    perclos: float = 1.0
    raw_weighted_score: float = 1.0


class SignalFusion:
    """
    Converts raw signals into normalised scores and fuses them.

    Usage:
        fusion = SignalFusion()
        components = fusion.compute(
            yaw=12.0, pitch=5.0,
            gaze_direction="center",
            ear=0.28,
            blink_rate=16.0,
            perclos=0.04,
        )
        print(f"Final score: {components.raw_weighted_score:.2f}")
        print(f"Head pose: {components.head_pose:.2f}")
    """

    def __init__(self):
        self._weights = cfg.weights
        # Validate weights sum to ~1.0
        total = sum(self._weights.values())
        if abs(total - 1.0) > 0.01:
            # Normalise
            for k in self._weights:
                self._weights[k] /= total

    def compute(self,
                yaw: float = 0.0,
                pitch: float = 0.0,
                gaze_direction: str = "center",
                ear: float = 0.30,
                blink_rate: float = 15.0,
                perclos: float = 0.05,
                ) -> ComponentScores:
        """
        Compute all component scores and fused score.

        Args:
            yaw:            Head yaw angle (degrees, 0=forward)
            pitch:          Head pitch angle (degrees, 0=forward)
            gaze_direction: Gaze label from GazeEstimator
            ear:            Eye Aspect Ratio (0.0-0.4)
            blink_rate:     Blinks per minute
            perclos:        Fraction of time eyes closed (0.0-1.0)

        Returns:
            ComponentScores with all individual + fused scores.
        """
        # 1. Head Pose Score
        hp_score = self._score_head_pose(yaw, pitch)

        # 2. Gaze Score
        gaze_score = self._score_gaze(gaze_direction)

        # 3. EAR Score
        ear_score = self._score_ear(ear)

        # 4. Blink Rate Score
        br_score = self._score_blink_rate(blink_rate)

        # 5. PERCLOS Score
        perclos_score = self._score_perclos(perclos)

        # 6. Weighted fusion
        fused = (
            hp_score * self._weights["head_pose"]
            + gaze_score * self._weights["gaze"]
            + ear_score * self._weights["ear"]
            + br_score * self._weights["blink_rate"]
            + perclos_score * self._weights["perclos"]
        )

        return ComponentScores(
            head_pose=round(hp_score, 4),
            gaze=round(gaze_score, 4),
            ear=round(ear_score, 4),
            blink_rate=round(br_score, 4),
            perclos=round(perclos_score, 4),
            raw_weighted_score=round(fused, 4),
        )

    # ──────────── Scoring Functions ────────────

    @staticmethod
    def _score_head_pose(yaw: float, pitch: float) -> float:
        """
        Score head pose: 1.0 (forward) → 0.0 (fully turned away).
        Uses smooth cosine decay for natural falloff.
        """
        # Yaw scoring
        abs_yaw = abs(yaw)
        if abs_yaw <= cfg.head_yaw_full_score:
            yaw_score = 1.0
        elif abs_yaw >= cfg.head_yaw_zero_score:
            yaw_score = 0.0
        else:
            # Smooth cosine interpolation
            t = (abs_yaw - cfg.head_yaw_full_score) / (
                cfg.head_yaw_zero_score - cfg.head_yaw_full_score)
            yaw_score = 0.5 * (1.0 + math.cos(math.pi * t))

        # Pitch scoring (same approach)
        abs_pitch = abs(pitch)
        if abs_pitch <= cfg.head_pitch_full_score:
            pitch_score = 1.0
        elif abs_pitch >= cfg.head_pitch_zero_score:
            pitch_score = 0.0
        else:
            t = (abs_pitch - cfg.head_pitch_full_score) / (
                cfg.head_pitch_zero_score - cfg.head_pitch_full_score)
            pitch_score = 0.5 * (1.0 + math.cos(math.pi * t))

        # Combined: take the lower of the two (most significant deviation)
        return min(yaw_score, pitch_score)

    @staticmethod
    def _score_gaze(direction: str) -> float:
        """Score gaze direction from lookup table."""
        return cfg.gaze_scores.get(direction.lower(), 0.3)

    @staticmethod
    def _score_ear(ear: float) -> float:
        """
        Score EAR: 1.0 (fully open) → 0.0 (closed).
        Linear interpolation between thresholds.
        """
        if ear >= cfg.ear_full_score:
            return 1.0
        elif ear <= cfg.ear_zero_score:
            return 0.0
        else:
            return (ear - cfg.ear_zero_score) / (
                cfg.ear_full_score - cfg.ear_zero_score)

    @staticmethod
    def _score_blink_rate(rate: float) -> float:
        """
        Score blink rate: optimal range = 1.0, deviations penalised.
        Normal: 12-20 blinks/min.
        """
        if cfg.blink_rate_optimal_low <= rate <= cfg.blink_rate_optimal_high:
            return 1.0

        if rate > cfg.blink_rate_optimal_high:
            # Too many blinks (fatigue)
            if rate >= cfg.blink_rate_penalty_high:
                return 0.2
            t = (rate - cfg.blink_rate_optimal_high) / (
                cfg.blink_rate_penalty_high - cfg.blink_rate_optimal_high)
            return max(0.2, 1.0 - t * 0.8)
        else:
            # Too few blinks (zoned out / staring)
            if rate <= cfg.blink_rate_penalty_low:
                return 0.4
            t = (cfg.blink_rate_optimal_low - rate) / (
                cfg.blink_rate_optimal_low - cfg.blink_rate_penalty_low)
            return max(0.4, 1.0 - t * 0.6)

    @staticmethod
    def _score_perclos(perclos: float) -> float:
        """
        Score PERCLOS: 0% closed = 1.0, >25% = 0.0.
        """
        if perclos <= cfg.perclos_full_score:
            return 1.0
        elif perclos >= cfg.perclos_zero_score:
            return 0.0
        else:
            t = (perclos - cfg.perclos_full_score) / (
                cfg.perclos_zero_score - cfg.perclos_full_score)
            return 1.0 - t