
"""
head_pose/pipeline.py
End-to-end Head Pose Estimation Pipeline.

Flow:
    Webcam Frame
        → MediaPipe Face Mesh (468/478 landmarks)
        → LandmarkMapper (extract 6 or 14 key points)
        → PoseCalculator (solvePnP → Euler angles)
        → DirectionClassifier (angles → labels)
        → AttentionJudge (direction → attention zone)
        → PoseVisualizer (draw overlays)
        → Annotated Frame

Usage:
    python -m head_pose
    # or
    from head_pose import HeadPosePipeline
    pipeline = HeadPosePipeline()
    pipeline.run()
"""
import cv2
import time
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass
from collections import deque

from .config import cfg
from .landmark_mapper import LandmarkMapper
from .pose_calculator import PoseCalculator, PoseResult
from .direction_classifier import DirectionClassifier, DirectionResult
from .attention_judge import AttentionJudge, AttentionResult
from .visualizer import PoseVisualizer


@dataclass
class HeadPoseResult:
    """Complete head pose result for one face."""
    face_index: int
    pose: PoseResult
    direction: DirectionResult
    attention: AttentionResult


class HeadPosePipeline:
    """
    Full head pose estimation pipeline with multi-face support.

    Usage:
        pipeline = HeadPosePipeline()

        # Option A: process single frame (for integration)
        results, annotated = pipeline.process_frame(bgr_frame)

        # Option B: live webcam loop
        pipeline.run()
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index

        # MediaPipe
        self._mp_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=cfg.max_faces,
            refine_landmarks=cfg.refine_landmarks,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

        # Per-face components (lazy init)
        self._mappers: Dict[int, LandmarkMapper] = {}
        self._calculators: Dict[int, PoseCalculator] = {}
        self._classifiers: Dict[int, DirectionClassifier] = {}
        self._judges: Dict[int, AttentionJudge] = {}
        self._visualizer = PoseVisualizer()

        # FPS
        self._fps_q = deque(maxlen=30)
        self._prev_time = time.perf_counter()

    def _get_components(self, face_idx: int, w: int, h: int):
        """Lazy-init per-face components."""
        if face_idx not in self._mappers:
            self._mappers[face_idx] = LandmarkMapper()
            self._calculators[face_idx] = PoseCalculator(w, h)
            self._classifiers[face_idx] = DirectionClassifier()
            self._judges[face_idx] = AttentionJudge()
        return (self._mappers[face_idx], self._calculators[face_idx],
                self._classifiers[face_idx], self._judges[face_idx])

    # ──────────── Single Frame API ────────────

    def process_frame(self, frame: np.ndarray) -> Tuple[List[HeadPoseResult], np.ndarray]:
        """
        Process one BGR frame.

        Returns:
            (list_of_HeadPoseResult, annotated_frame)
        """
        h, w, _ = frame.shape

        # Resize for performance
        if w > cfg.process_width:
            ratio = cfg.process_width / w
            frame = cv2.resize(frame, (cfg.process_width, int(h * ratio)))
            h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        all_results: List[HeadPoseResult] = []

        if results.multi_face_landmarks:
            for idx, face_lm in enumerate(results.multi_face_landmarks):
                lm = face_lm.landmark
                mapper, calc, classifier, judge = self._get_components(idx, w, h)

                # 1. Map landmarks
                mapped = mapper.extract(lm, w, h)

                # 2. Compute pose
                pose = calc.compute(mapped)
                if pose is None:
                    continue

                # 3. Classify direction
                direction = classifier.classify(pose)

                # 4. Judge attention
                attention = judge.evaluate(pose)

                result = HeadPoseResult(
                    face_index=idx,
                    pose=pose,
                    direction=direction,
                    attention=attention,
                )
                all_results.append(result)

                # 5. Visualise
                contour = mapper.extract_all_contour(lm, w, h) if cfg.draw_face_wireframe else None
                frame = self._visualizer.draw(
                    frame, pose, direction, attention,
                    face_contour=contour,
                    mapped_points=mapped.all_points_2d,
                    face_index=idx,
                )

        # FPS
        now = time.perf_counter()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            self._fps_q.append(1.0 / dt)
        fps = sum(self._fps_q) / len(self._fps_q) if self._fps_q else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Face count
        cv2.putText(frame, f"Faces: {len(all_results)}", (w - 130, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return all_results, frame

    # ──────────── Live Webcam Loop ────────────

    def run(self, on_result: Optional[Callable] = None):
        """
        Run live webcam head pose estimation with display.

        Args:
            on_result: Optional callback(results, frame) per processed frame.

        Keys:
            q/ESC  → Quit
            a      → Toggle 3D axes
            d      → Toggle direction labels
            g      → Toggle angle gauges
            w      → Toggle wireframe
            p      → Toggle landmark points
            r      → Reset all trackers
        """
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("ERROR: Cannot open webcam.")
            return

        print("=" * 60)
        print("  Head Pose Estimation — Live Demo")
        print("=" * 60)
        print(f"  Camera       : {self.camera_index}")
        print(f"  Resolution   : {int(cap.get(3))}x{int(cap.get(4))}")
        print(f"  Model        : {LandmarkMapper().mode} ({LandmarkMapper().point_count} points)")
        print(f"  solvePnP     : method={cfg.solvepnp_method}")
        print(f"  Kalman filter: {'ON' if cfg.use_kalman else 'OFF'}")
        print(f"  Teacher pos  : yaw={cfg.teacher_yaw}°, pitch={cfg.teacher_pitch}°")
        print("-" * 60)
        print("  Keys: [q]uit [a]xes [d]irection [g]auges [w]ire [p]oints [r]eset")
        print("=" * 60)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pose_results, annotated = self.process_frame(frame)

            if on_result and pose_results:
                on_result(pose_results, annotated)

            cv2.imshow("Head Pose Estimation", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('a'):
                cfg.draw_3d_axes = not cfg.draw_3d_axes
                print(f"  3D axes: {'ON' if cfg.draw_3d_axes else 'OFF'}")
            elif key == ord('d'):
                cfg.draw_direction_label = not cfg.draw_direction_label
                print(f"  Direction labels: {'ON' if cfg.draw_direction_label else 'OFF'}")
            elif key == ord('g'):
                cfg.draw_angle_gauges = not cfg.draw_angle_gauges
                print(f"  Angle gauges: {'ON' if cfg.draw_angle_gauges else 'OFF'}")
            elif key == ord('w'):
                cfg.draw_face_wireframe = not cfg.draw_face_wireframe
                print(f"  Wireframe: {'ON' if cfg.draw_face_wireframe else 'OFF'}")
            elif key == ord('p'):
                cfg.draw_landmark_points = not cfg.draw_landmark_points
                print(f"  Landmark points: {'ON' if cfg.draw_landmark_points else 'OFF'}")
            elif key == ord('r'):
                self.reset()
                print("  All trackers reset.")

        cap.release()
        cv2.destroyAllWindows()
        self.release()
        print("\nShutdown complete.")

    def reset(self):
        """Reset all per-face components."""
        for c in self._calculators.values():
            c.reset()
        for j in self._judges.values():
            j.reset()

    def release(self):
        self._face_mesh.close()

