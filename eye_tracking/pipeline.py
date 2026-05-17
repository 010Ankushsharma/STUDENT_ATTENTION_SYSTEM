
"""
eye_tracking/pipeline.py
End-to-end Eye Tracking Pipeline — ties all components together.

Usage:
    # Standalone demo:
    python -m eye_tracking

    # Integration:
    from eye_tracking import EyeTrackingPipeline
    pipeline = EyeTrackingPipeline()
    results = pipeline.process_frame(frame)
"""
import cv2
import time
import numpy as np
import mediapipe as mp
from typing import Tuple, Dict, Optional, Callable, Any
from collections import deque
from dataclasses import dataclass

from .config import cfg
from .ear_calculator import EARCalculator
from .blink_detector import BlinkDetector, BlinkMetrics
from .drowsiness_detector import DrowsinessDetector, DrowsinessResult
from .gaze_estimator import GazeEstimator, GazeResult
from .visualizer import EyeVisualizer


@dataclass
class EyeTrackingResult:
    """Complete eye tracking result for one face in one frame."""
    face_index: int
    ear_left: float
    ear_right: float
    ear_avg: float
    blink_metrics: BlinkMetrics
    drowsiness: DrowsinessResult
    gaze: GazeResult
    landmarks: Any = None


class EyeTrackingPipeline:
    """
    Full eye tracking pipeline for multiple faces.

    Components:
        MediaPipe Face Mesh → EAR → Blink → Drowsiness → Gaze → Visualise

    Usage:
        pipeline = EyeTrackingPipeline()

        # Option A: Process single frame
        results, annotated = pipeline.process_frame(bgr_frame)

        # Option B: Live webcam loop
        pipeline.run()
    """

    def __init__(self, camera_index: int = 0, fps: float = 30.0):
        self.camera_index = camera_index

        # MediaPipe Face Mesh
        self._mp_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=cfg.max_faces,
            refine_landmarks=cfg.refine_landmarks,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

        # Per-face component instances (lazy-created)
        self._ear_calcs: Dict[int, EARCalculator] = {}
        self._blink_dets: Dict[int, BlinkDetector] = {}
        self._drowsy_dets: Dict[int, DrowsinessDetector] = {}
        self._gaze_est = GazeEstimator()
        self._visualizer = EyeVisualizer()

        # FPS tracking
        self._fps_q = deque(maxlen=30)
        self._prev_time = time.perf_counter()
        self._frame_num = 0

    def _get_components(self, face_idx: int):
        """Lazy-init per-face components."""
        if face_idx not in self._ear_calcs:
            self._ear_calcs[face_idx] = EARCalculator()
            self._blink_dets[face_idx] = BlinkDetector()
            self._drowsy_dets[face_idx] = DrowsinessDetector()
        return (self._ear_calcs[face_idx],
                self._blink_dets[face_idx],
                self._drowsy_dets[face_idx])

    # ──────────── Single Frame API ────────────

    def process_frame(self, frame: np.ndarray
                      ) -> Tuple[list, np.ndarray]:
        """
        Process one BGR frame.

        Returns:
            (list_of_EyeTrackingResult, annotated_frame)
        """
        self._frame_num += 1
        h, w, _ = frame.shape

        # Resize for speed
        if w > cfg.process_width:
            ratio = cfg.process_width / w
            frame = cv2.resize(frame, (cfg.process_width, int(h * ratio)))
            h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        all_results = []

        if results.multi_face_landmarks:
            for idx, face_lm in enumerate(results.multi_face_landmarks):
                lm = face_lm.landmark
                ear_calc, blink_det, drowsy_det = self._get_components(idx)

                # 1. EAR
                ear_l, ear_r, ear_avg = ear_calc.compute(lm, w, h)

                # 2. Blink detection
                blink_metrics = blink_det.update(ear_avg, self._frame_num)

                # 3. Drowsiness
                drowsiness = drowsy_det.assess(blink_metrics)

                # 4. Gaze
                gaze = self._gaze_est.estimate(lm, w, h)

                result = EyeTrackingResult(
                    face_index=idx,
                    ear_left=ear_l,
                    ear_right=ear_r,
                    ear_avg=ear_avg,
                    blink_metrics=blink_metrics,
                    drowsiness=drowsiness,
                    gaze=gaze,
                    landmarks=lm,
                )
                all_results.append(result)

                # 5. Visualise
                frame = self._visualizer.draw(
                    frame, lm, w, h,
                    blink_metrics=blink_metrics,
                    drowsiness=drowsiness,
                    gaze=gaze,
                )

        # FPS overlay
        now = time.perf_counter()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            self._fps_q.append(1.0 / dt)
        fps = sum(self._fps_q) / len(self._fps_q) if self._fps_q else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        return all_results, frame

    # ──────────── Live Webcam Loop ────────────

    def run(self, on_result: Optional[Callable] = None):
        """
        Run live webcam eye tracking with display.

        Args:
            on_result: Optional callback(results, frame) per frame.

        Keys:
            q/ESC → quit    r → reset    c → toggle contours
            i → toggle iris    m → toggle metrics panel
        """
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("ERROR: Cannot open webcam.")
            return

        print("=" * 58)
        print("  Eye Tracking & Blink Detection — Live Demo")
        print("=" * 58)
        print(f"  Camera      : {self.camera_index}")
        print(f"  Resolution  : {int(cap.get(3))}x{int(cap.get(4))}")
        print(f"  Iris tracking: {'ON' if cfg.refine_landmarks else 'OFF'}")
        print(f"  EAR threshold: {cfg.ear_blink_threshold}")
        print("-" * 58)
        print("  Keys: [q]uit [r]eset [c]ontours [i]ris [m]etrics")
        print("=" * 58)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tracking_results, annotated = self.process_frame(frame)

            if on_result and tracking_results:
                on_result(tracking_results, annotated)

            cv2.imshow("Eye Tracking & Blink Detection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                self.reset()
                print("  All detectors reset.")
            elif key == ord('c'):
                cfg.draw_eye_contours = not cfg.draw_eye_contours
                print(f"  Eye contours: {'ON' if cfg.draw_eye_contours else 'OFF'}")
            elif key == ord('i'):
                cfg.draw_iris = not cfg.draw_iris
                print(f"  Iris markers: {'ON' if cfg.draw_iris else 'OFF'}")
            elif key == ord('m'):
                cfg.draw_metrics_panel = not cfg.draw_metrics_panel
                print(f"  Metrics panel: {'ON' if cfg.draw_metrics_panel else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()
        self.release()
        print("\nShutdown complete.")

    def reset(self):
        """Reset all per-face detectors."""
        for c in self._ear_calcs.values():
            c.reset()
        for b in self._blink_dets.values():
            b.reset()
        for d in self._drowsy_dets.values():
            d.reset()

    def release(self):
        """Free resources."""
        self._face_mesh.close()

