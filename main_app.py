
"""
main_app.py
═══════════════════════════════════════════════════════════════
  STUDENT ATTENTION DETECTION SYSTEM — Final Integrated Build
═══════════════════════════════════════════════════════════════

Modules Integrated:
    1. Face Detection       (face_detection/)
    2. Eye Tracking         (eye_tracking/)
    3. Blink Detection      (eye_tracking/blink_detector.py)
    4. Head Pose Estimation (head_pose/)
    5. Attention Scoring    (attention_scoring/)
    6. Database             (database/)
    7. Web Dashboard        (dashboard/)
    + Core Utilities        (core/)

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    MAIN PIPELINE                          │
    │                                                          │
    │  Webcam → Preprocess → Face Detect → Track IDs           │
    │                              │                            │
    │              ┌───────────────┼───────────────┐           │
    │              ▼               ▼               ▼           │
    │         Eye Track      Head Pose       (per student)     │
    │         • EAR          • solvePnP                        │
    │         • Blinks       • Direction                       │
    │         • Gaze         • Attention                       │
    │         • Drowsiness   • Kalman                          │
    │              │               │                            │
    │              └───────┬───────┘                           │
    │                      ▼                                    │
    │              Attention Scoring                            │
    │              • Signal Fusion                              │
    │              • EMA Smoothing                              │
    │              • State Classification                       │
    │              • Alert Generation                           │
    │                      │                                    │
    │           ┌──────────┼──────────┐                       │
    │           ▼          ▼          ▼                        │
    │      Database    Dashboard   Visualizer                  │
    │      (SQLite)   (FastAPI)   (OpenCV)                     │
    └─────────────────────────────────────────────────────────┘

Usage:
    python main_app.py
    python main_app.py --camera 1 --no-display --log
    python main_app.py --dashboard --port 8080

Performance Targets:
    • 25-30 FPS with 1-3 students
    • 15-20 FPS with 5-10 students
    • <50ms per frame total latency
"""
import cv2
import time
import csv
import argparse
import numpy as np
import mediapipe as mp
import threading
from collections import deque
from typing import Dict, Optional, Any

# ── Core utilities ──
from core import (
    get_logger, setup_logging,
    PerformanceMonitor, FPSCounter, Timer,
    CameraError, ProcessingError,
    load_config_from_env,
)

# ── Module imports ──
from face_detection.detector import FaceDetector
from face_detection.tracker import CentroidTracker
from face_detection.preprocessor import FramePreprocessor
from face_detection.annotator import FrameAnnotator

from eye_tracking.ear_calculator import EARCalculator
from eye_tracking.blink_detector import BlinkDetector
from eye_tracking.drowsiness_detector import DrowsinessDetector
from eye_tracking.gaze_estimator import GazeEstimator
from eye_tracking.visualizer import EyeVisualizer

from head_pose.landmark_mapper import LandmarkMapper
from head_pose.pose_calculator import PoseCalculator
from head_pose.direction_classifier import DirectionClassifier
from head_pose.attention_judge import AttentionJudge
from head_pose.visualizer import PoseVisualizer

from attention_scoring.pipeline import ScoringPipeline
from attention_scoring.state_classifier import AttentionState

from database import DatabaseManager, LiveLogger, AnalyticsEngine

from main_config import sys_cfg

# ── Optional dashboard ──
try:
    from dashboard import DashboardServer
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

# ── Logger ──
logger = get_logger("main")


# ═══════════════════════════════════════════════════════════
# COLOUR CONSTANTS
# ═══════════════════════════════════════════════════════════
STATE_COLORS = {
    "attentive":    (0, 200, 0),
    "distracted":   (0, 200, 255),
    "sleepy":       (0, 0, 255),
    "looking_away": (255, 100, 0),
}
WHITE = (255, 255, 255)
DARK = (30, 30, 30)
GRAY = (160, 160, 160)


# ═══════════════════════════════════════════════════════════
# PER-STUDENT PROCESSOR
# ═══════════════════════════════════════════════════════════
class StudentProcessor:
    """Holds per-student instances of all sub-module processors."""

    __slots__ = [
        "student_id", "ear_calc", "blink_det", "drowsy_det",
        "gaze_est", "eye_vis", "lm_mapper", "pose_calc",
        "dir_clf", "attn_judge", "pose_vis",
    ]

    def __init__(self, student_id: int, frame_w: int, frame_h: int):
        self.student_id = student_id
        self.ear_calc = EARCalculator()
        self.blink_det = BlinkDetector(fps=sys_cfg.target_fps)
        self.drowsy_det = DrowsinessDetector()
        self.gaze_est = GazeEstimator()
        self.eye_vis = EyeVisualizer()
        self.lm_mapper = LandmarkMapper(use_extended=True)
        self.pose_calc = PoseCalculator(frame_w, frame_h)
        self.dir_clf = DirectionClassifier()
        self.attn_judge = AttentionJudge()
        self.pose_vis = PoseVisualizer()


# ═══════════════════════════════════════════════════════════
# DASHBOARD DRAWER (on-screen overlay)
# ═══════════════════════════════════════════════════════════
class DashboardDrawer:
    """Draws the on-screen dashboard overlay."""

    @staticmethod
    def draw(frame: np.ndarray, scoring: ScoringPipeline,
             fps: float, frame_count: int) -> np.ndarray:
        h, w = frame.shape[:2]
        summary = scoring.get_class_summary()

        panel_h = 110
        panel_y = h - panel_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, h), DARK, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.line(frame, (0, panel_y), (w, panel_y), GRAY, 1)

        cv2.putText(frame, "CLASSROOM DASHBOARD",
                    (15, panel_y + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 200, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, f"FPS: {fps:.1f}  |  Frame: {frame_count}",
                    (w - 250, panel_y + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, GRAY, 1, cv2.LINE_AA)

        box_y = panel_y + 35
        box_w, box_h = 135, 28
        states = [
            ("ATTENTIVE", summary.get("attentive", 0), STATE_COLORS["attentive"]),
            ("DISTRACTED", summary.get("distracted", 0), STATE_COLORS["distracted"]),
            ("SLEEPY", summary.get("sleepy", 0), STATE_COLORS["sleepy"]),
            ("LOOKING AWAY", summary.get("looking_away", 0), STATE_COLORS["looking_away"]),
        ]

        x = 15
        for label, count, color in states:
            cv2.rectangle(frame, (x, box_y), (x + box_w, box_y + box_h), color, -1)
            cv2.putText(frame, f"{label}: {count}", (x + 6, box_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
            x += box_w + 10

        avg = summary.get("avg_score", 1.0)
        bar_x, bar_y = 15, box_y + box_h + 12
        bar_w, bar_h = w - 30, 18
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), GRAY, 1)
        fill_w = int(bar_w * avg)
        bar_color = (STATE_COLORS["attentive"] if avg >= 0.7
                     else STATE_COLORS["distracted"] if avg >= 0.45
                     else STATE_COLORS["sleepy"])
        cv2.rectangle(frame, (bar_x + 1, bar_y + 1),
                      (bar_x + fill_w, bar_y + bar_h - 1), bar_color, -1)
        cv2.putText(frame, f"Class Avg: {avg:.0%}", (bar_x + 8, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Students: {summary.get('total', 0)}",
                    (bar_x + bar_w - 110, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_student_badge(frame, bbox, student_id, score, state):
        x1, y1, x2, y2 = bbox
        color = STATE_COLORS.get(state, GRAY)
        bar_y = y2 + 5
        bar_w, bar_h = x2 - x1, 8
        cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h), DARK, -1)
        cv2.rectangle(frame, (x1, bar_y), (x1 + int(bar_w * score), bar_y + bar_h), color, -1)
        cv2.putText(frame, f"S{student_id}: {state.upper()} {int(score*100)}%",
                    (x1, bar_y + bar_h + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
        return frame


# ═══════════════════════════════════════════════════════════
# MAIN APPLICATION CLASS
# ═══════════════════════════════════════════════════════════
class StudentAttentionSystem:
    """
    The complete Student Attention Detection System.

    Integrates all 7 modules with error handling, performance
    monitoring, and deployment-ready architecture.
    """

    def __init__(self, camera_index: int = 0, show_video: bool = True,
                 enable_dashboard: bool = True):
        logger.info("Initializing Student Attention System...")

        self.camera_index = camera_index
        self.show_video = show_video
        self.enable_dashboard = enable_dashboard and DASHBOARD_AVAILABLE

        # ── Performance Monitoring ──
        self.perf = PerformanceMonitor()
        self.fps = FPSCounter()

        # ── Face Detection ──
        logger.info("Loading face detection...")
        self.preprocessor = FramePreprocessor()
        self.face_detector = FaceDetector(use_mesh=True)
        self.face_tracker = CentroidTracker()
        self.face_annotator = FrameAnnotator()

        # ── MediaPipe Face Mesh ──
        logger.info("Loading MediaPipe face mesh...")
        self._mp_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=15,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ── Attention Scoring ──
        self.scoring = ScoringPipeline()

        # ── Per-student processors ──
        self._processors: Dict[int, StudentProcessor] = {}

        # ── Dashboard overlay ──
<<<<<<< HEAD
       # self.dashboard_drawer = DashboardDrawer()
        from dashboard.clean_overlay import CleanOverlay
        self.dashboard_drawer = CleanOverlay()
=======
        self.dashboard_drawer = DashboardDrawer()
>>>>>>> c9284a3ad6c0217a589474a09ab81a46493769a6

        # ── Database ──
        logger.info("Initializing database...")
        try:
            self._db = DatabaseManager()
            self._logger = LiveLogger(self._db)
            self._analytics = AnalyticsEngine(self._db)
            self._db_available = True
            logger.info(f"Database ready: {self._db.get_db_size_mb():.2f} MB")
        except Exception as e:
            logger.warning(f"Database init failed: {e}. Running without DB.")
            self._db = None
            self._logger = None
            self._analytics = None
            self._db_available = False

        # ── Web Dashboard ──
        self._dashboard: Optional[Any] = None
        if self.enable_dashboard:
            try:
                self._dashboard = DashboardServer(
                    scoring=self.scoring,
                    db=self._db,
                    analytics=self._analytics,
                    logger=self._logger,
                )
                logger.info("Dashboard server ready")
            except Exception as e:
                logger.warning(f"Dashboard init failed: {e}")
                self._dashboard = None

        # ── State ──
        self._frame_count = 0
        self._session_id = None
        self._alert_flash_until = 0.0

        # ── Alert callback ──
        self.scoring.set_alert_callback(self._on_alert)

        logger.info("✅ System initialization complete")

    def _get_processor(self, student_id: int, w: int, h: int) -> StudentProcessor:
        if student_id not in self._processors:
            self._processors[student_id] = StudentProcessor(student_id, w, h)
        return self._processors[student_id]

    def _on_alert(self, alert):
        """Handle alert — log, DB, visual flash."""
        logger.warning(f"ALERT: {alert.message} (student {alert.student_id})")
        if sys_cfg.alert_visual:
            self._alert_flash_until = time.time() + 1.5
        if self._db_available and self._session_id:
            try:
                self._logger.log_alert(
                    student_id=alert.student_id,
                    alert_type=alert.alert_type.value,
                    severity=alert.severity.value,
                    message=alert.message,
                    score=alert.score,
                    state=alert.state.value if hasattr(alert.state, "value") else str(alert.state),
                    sustained_frames=getattr(alert, "sustained_frames", 0),
                )
            except Exception as e:
                logger.error(f"Alert DB write failed: {e}")

    # ──────────────────────────────────────────────────────
    # CORE: Process a single frame
    # ──────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one frame through the complete pipeline.

        Pipeline stages (with timing):
            1. Preprocess      (~2ms)
            2. Face Detection  (~8ms)
            3. Face Mesh       (~12ms)
            4. Eye Tracking    (~3ms per student)
            5. Head Pose       (~2ms per student)
            6. Scoring         (~1ms per student)
            7. Visualization   (~3ms)
            8. DB Logging      (~0.1ms, batched)
        """
        self._frame_count += 1
        self.fps.tick()

        # ── 1. Preprocess ──
        with Timer("preprocess") as t:
            processed = self.preprocessor.process(frame)
        self.perf.record("preprocess", t.elapsed_ms)
        ph, pw = processed.shape[:2]

        # ── 2. Face Detection + Tracking ──
        with Timer("face_detect") as t:
            faces = self.face_detector.detect(processed)
            students = self.face_tracker.update(faces)
        self.perf.record("face_detect", t.elapsed_ms)

        # ── 3. MediaPipe Face Mesh ──
        with Timer("face_mesh") as t:
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            mesh_results = self._face_mesh.process(rgb)
        self.perf.record("face_mesh", t.elapsed_ms)

        mesh_faces = (mesh_results.multi_face_landmarks
                      if mesh_results and mesh_results.multi_face_landmarks
                      else [])

        # ── 4-6. Per-student processing ──
        with Timer("per_student") as t:
            for sid, student in students.items():
                if student.disappeared > 0:
                    continue
                self._process_student(sid, student, mesh_faces, processed, pw, ph)
        self.perf.record("per_student", t.elapsed_ms)

        # ── 7. Visualization ──
        with Timer("visualization") as t:
            current_fps = self.fps.get()
            processed = self.face_annotator.draw(processed, students, fps=current_fps)

            if sys_cfg.show_dashboard:
                processed = self.dashboard_drawer.draw(
                    processed, self.scoring, current_fps, self._frame_count
                )

            if sys_cfg.alert_visual and time.time() < self._alert_flash_until:
                cv2.rectangle(processed, (0, 0), (pw - 1, ph - 1), (0, 0, 255), 4)
        self.perf.record("visualization", t.elapsed_ms)

        # ── 8. Push to dashboard ──
        if self._dashboard:
            self._dashboard.update_frame(processed)

        return processed

    def _process_student(self, sid, student, mesh_faces, frame, pw, ph):
        """Process all modules for a single student."""
        proc = self._get_processor(sid, pw, ph)

        best_mesh = self._match_mesh(student.centroid, mesh_faces, pw, ph)

        # Defaults
        ear_avg, blink_rate, perclos = 0.30, 15.0, 0.05
        gaze_dir, drowsy_level = "center", "alert"
        yaw, pitch, head_dir = 0.0, 0.0, "forward"
        hp_score = gaze_score = ear_score = blink_score = perc_score = 0.0

        if best_mesh is not None:
            lm = best_mesh.landmark

            # ── Eye Tracking + Blink Detection ──
            try:
                _, _, ear_avg = proc.ear_calc.compute(lm, pw, ph)
                blink_metrics = proc.blink_det.update(ear_avg, self._frame_count)
                drowsy_result = proc.drowsy_det.assess(blink_metrics)
                gaze_result = proc.gaze_est.estimate(lm, pw, ph)

                blink_rate = blink_metrics.blink_rate_per_min
                perclos = blink_metrics.perclos
                gaze_dir = gaze_result.direction_label
                drowsy_level = drowsy_result.level.value

<<<<<<< HEAD
                # frame = proc.eye_vis.draw(frame, lm, pw, ph,
                #                           blink_metrics=blink_metrics,
                #                           drowsiness=drowsy_result,
                #                           gaze=gaze_result)
=======
                frame = proc.eye_vis.draw(frame, lm, pw, ph,
                                          blink_metrics=blink_metrics,
                                          drowsiness=drowsy_result,
                                          gaze=gaze_result)
>>>>>>> c9284a3ad6c0217a589474a09ab81a46493769a6
            except Exception as e:
                logger.debug(f"Eye tracking error S{sid}: {e}")

            # ── Head Pose Estimation ──
            try:
                mapped = proc.lm_mapper.extract(lm, pw, ph)
                pose = proc.pose_calc.compute(mapped)
                if pose:
                    direction = proc.dir_clf.classify(pose)
                    attention = proc.attn_judge.evaluate(pose)
                    yaw, pitch = pose.yaw, pose.pitch
                    head_dir = direction.combined_label
<<<<<<< HEAD
                    # frame = proc.pose_vis.draw(frame, pose, direction,
                    #                           attention, face_index=sid)
=======
                    frame = proc.pose_vis.draw(frame, pose, direction,
                                              attention, face_index=sid)
>>>>>>> c9284a3ad6c0217a589474a09ab81a46493769a6
            except Exception as e:
                logger.debug(f"Head pose error S{sid}: {e}")

        # ── Attention Scoring ──
        try:
            result = self.scoring.update(
                student_id=sid, ear=ear_avg, gaze_direction=gaze_dir,
                yaw=yaw, pitch=pitch, blink_rate=blink_rate,
                perclos=perclos, drowsiness_level=drowsy_level,
                head_direction=head_dir,
            )

            # DB logging (batched)
            if self._db_available and self._session_id:
                components = getattr(result, "components", {})
                self._logger.log_score(
                    student_id=sid, frame_num=self._frame_count,
                    score=result.score, state=result.state,
                    ear=ear_avg, blink_rate=blink_rate, perclos=perclos,
                    gaze_direction=gaze_dir, drowsiness=drowsy_level,
                    yaw=yaw, pitch=pitch, head_direction=head_dir,
                    hp_score=components.get("head_pose", 0),
                    gaze_score=components.get("gaze", 0),
                    ear_score=components.get("ear", 0),
                    blink_score=components.get("blink_rate", 0),
                    perclos_score=components.get("perclos", 0),
                )

            self.dashboard_drawer.draw_student_badge(
                frame, student.bbox, sid, result.score, result.state
            )
        except Exception as e:
            logger.error(f"Scoring error S{sid}: {e}")

    def _match_mesh(self, centroid, mesh_faces, w, h):
        """Match closest mesh face to tracked student centroid."""
        if not mesh_faces:
            return None
        cx, cy = centroid
        best, best_dist = None, float("inf")
        for face_lm in mesh_faces:
            nose = face_lm.landmark[1]
            dist = ((cx - nose.x * w) ** 2 + (cy - nose.y * h) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best = dist, face_lm
        return best if best_dist < 150 else None

    # ──────────────────────────────────────────────────────
    # RUN: Main loop
    # ──────────────────────────────────────────────────────
    def run(self):
        """Run the complete system with live webcam."""
        # ── Load env config ──
        try:
            load_config_from_env()
        except Exception:
            pass

        # ── Open camera ──
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, sys_cfg.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, sys_cfg.camera_height)
        cap.set(cv2.CAP_PROP_FPS, sys_cfg.target_fps)

        if not cap.isOpened():
            raise CameraError(
                f"Cannot open camera index {self.camera_index}",
                camera_index=self.camera_index,
            )

        # ── Start DB session ──
        if self._db_available:
            self._session_id = self._logger.start_session(
                name=sys_cfg.session_name,
                camera_index=self.camera_index,
                config={"process_width": sys_cfg.process_width,
                        "target_fps": sys_cfg.target_fps},
            )
            logger.info(f"DB session started: {self._session_id}")

        # ── Start dashboard ──
        if self._dashboard:
            self._dashboard.start(port=sys_cfg.dashboard_port)

        self._print_banner(cap)

        # ── Main loop ──
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame capture failed, retrying...")
                    time.sleep(0.1)
                    continue

                # Skip frames for performance
                if sys_cfg.skip_frames > 0:
                    if self._frame_count % (sys_cfg.skip_frames + 1) != 0:
                        self._frame_count += 1
                        continue

                with Timer("total_frame") as t:
                    annotated = self.process_frame(frame)
                self.perf.record("total_frame", t.elapsed_ms)

                if self.show_video:
                    cv2.imshow(sys_cfg.window_name, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("d"):
                    sys_cfg.show_dashboard = not sys_cfg.show_dashboard
                elif key == ord("s"):
                    cv2.imwrite(f"screenshot_{self._frame_count}.jpg", annotated)
                    logger.info(f"Screenshot saved: screenshot_{self._frame_count}.jpg")
                elif key == ord("r"):
                    self._reset()
                elif key == ord("p"):
                    self._print_summary()
                elif key == ord("a"):
                    self._print_analytics()
                elif key == ord("w"):
                    import webbrowser
                    webbrowser.open(f"http://localhost:{sys_cfg.dashboard_port}")
                elif key == ord("m"):
                    self.perf.print_report()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            self._shutdown(cap)

    def _shutdown(self, cap):
        """Clean shutdown of all components."""
        logger.info("Shutting down...")

        cap.release()
        cv2.destroyAllWindows()

        # Face mesh
        try:
            self._face_mesh.close()
        except Exception:
            pass

        # Dashboard
        if self._dashboard:
            self._dashboard.stop()

        # Database
        if self._db_available and self._session_id:
            summary = self.scoring.get_class_summary()
            self._logger.end_session(
                total_frames=self._frame_count,
                avg_score=summary.get("avg_score", 0),
                total_students=summary.get("total", 0),
            )
            logger.info(f"DB session ended: {self._session_id}")

        if self._db:
            self._db.close()

        # Performance report
        self.perf.print_report()
        budget = self.perf.get_total_frame_budget(sys_cfg.target_fps)
        logger.info(f"Frame budget: {budget['used_ms']:.1f}/{budget['budget_ms']:.1f}ms "
                    f"({budget['utilization_pct']:.0f}%)")

        self._print_final_report()
        logger.info("✅ Shutdown complete")

    def _reset(self):
        self.face_tracker.reset()
        self.scoring.reset()
        self._processors.clear()
        logger.info("All trackers reset")

    def _print_banner(self, cap):
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print()
        print("  ╔═══════════════════════════════════════════════════════════╗")
        print("  ║   STUDENT ATTENTION DETECTION SYSTEM  v1.0.0             ║")
        print("  ╠═══════════════════════════════════════════════════════════╣")
        print(f"  ║  Camera    : {self.camera_index} ({w}x{h})")
        print(f"  ║  Process   : {sys_cfg.process_width}px | FPS target: {sys_cfg.target_fps}")
        print(f"  ║  Dashboard : {'http://localhost:' + str(sys_cfg.dashboard_port) if self._dashboard else 'OFF'}")
        print(f"  ║  Database  : {'ON' if self._db_available else 'OFF'}")
        print(f"  ║  Session   : {self._session_id or 'N/A'}")
        print("  ╠═══════════════════════════════════════════════════════════╣")
        print("  ║  Keys: [q]uit [d]ashboard [s]creenshot [r]eset          ║")
        print("  ║        [p]rint [a]nalytics [w]eb [m]etrics              ║")
        print("  ╚═══════════════════════════════════════════════════════════╝")
        print()

    def _print_summary(self):
        summary = self.scoring.get_class_summary()
        print(f"\n  CLASS: {summary['total']} students | "
              f"Avg: {summary.get('avg_score', 0):.0%} | "
              f"🟢{summary.get('attentive',0)} "
              f"🟡{summary.get('distracted',0)} "
              f"🔴{summary.get('sleepy',0)} "
              f"🔵{summary.get('looking_away',0)}\n")

    def _print_analytics(self):
        if not self._db_available or not self._session_id:
            print("  No DB session active")
            return
        self._logger.flush()
        overview = self._analytics.session_overview(self._session_id)
        print(f"\n  📊 Records: {overview.get('total_score_records', 0)} | "
              f"Avg: {overview.get('avg_score', 0):.2f} | "
              f"Alerts: {overview.get('alert_count', 0)}\n")

    def _print_final_report(self):
        summary = self.scoring.get_class_summary()
        alerts = self.scoring.get_alerts()
        print()
        print("  ════════════ FINAL SESSION REPORT ════════════")
        print(f"  Frames processed : {self._frame_count}")
        print(f"  Students tracked : {summary.get('total', 0)}")
        print(f"  Class avg score  : {summary.get('avg_score', 0):.0%}")
        print(f"  Alerts generated : {len(alerts)}")
        if self._db_available:
            counts = self._db.get_table_counts()
            print(f"  DB records       : {counts.get('attention_scores', 0)}")
            print(f"  DB size          : {self._db.get_db_size_mb():.2f} MB")
        print("  ═══════════════════════════════════════════════")
        print()


# ═══════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Student Attention Detection System v1.0.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_app.py                     # Default: camera 0, dashboard on
  python main_app.py --camera 1          # Use camera 1
  python main_app.py --no-display --log  # Headless with CSV logging
  python main_app.py --port 8080         # Dashboard on port 8080
  python main_app.py --no-dashboard      # Disable web dashboard
        """,
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--skip-frames", type=int, default=0)
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_to_file=args.log)

    # Apply CLI args to config
    sys_cfg.process_width = args.width
    sys_cfg.log_to_file = args.log
    sys_cfg.dashboard_port = args.port
    sys_cfg.skip_frames = args.skip_frames

    # Run
    system = StudentAttentionSystem(
        camera_index=args.camera,
        show_video=not args.no_display,
        enable_dashboard=not args.no_dashboard,
    )
    system.run()


if __name__ == "__main__":
    main()
