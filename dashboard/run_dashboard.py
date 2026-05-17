
"""
dashboard/run_dashboard.py
Standalone script to run the dashboard with the full system.

Usage:
    python -m dashboard.run_dashboard
    python -m dashboard.run_dashboard --port 8080 --camera 1
"""
import argparse
import cv2
import time
import numpy as np
import threading
from collections import deque

# System imports
from attention_scoring.pipeline import ScoringPipeline
from database import DatabaseManager, LiveLogger, AnalyticsEngine
from dashboard import DashboardServer
from dashboard.config import dashboard_cfg


def main():
    parser = argparse.ArgumentParser(description="Attention Dashboard + System")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args()

    # ── Init components ──
    scoring = ScoringPipeline()
    db = DatabaseManager()
    logger = LiveLogger(db)
    analytics = AnalyticsEngine(db)

    # ── Start dashboard server ──
    server = DashboardServer(
        scoring=scoring,
        db=db,
        analytics=analytics,
        logger=logger,
    )
    server.start(port=args.port)

    # ── Start session ──
    session_id = logger.start_session(
        name="Dashboard Session",
        camera_index=args.camera,
    )
    print(f"  📝 Session: {session_id}")

    # ── Alert callback ──
    def on_alert(alert):
        logger.log_alert(
            student_id=alert.student_id,
            alert_type=alert.alert_type.value,
            severity=alert.severity.value,
            message=alert.message,
            score=alert.score,
            state=alert.state.value if hasattr(alert.state, 'value') else str(alert.state),
        )
        print(f"  🚨 {alert.message}")

    scoring.set_alert_callback(on_alert)

    # ── Open camera ──
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    print(f"\n  🎥 Camera {args.camera} opened")
    print(f"  🌐 Dashboard: http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    # ── Import the full processing system ──
    try:
        from main_app import StudentAttentionSystem
        system = StudentAttentionSystem(camera_index=args.camera, show_video=True)
        use_full_system = True
        print("  ✅ Full processing system loaded")
    except ImportError:
        use_full_system = False
        print("  ⚠ Running in demo mode (install mediapipe for full system)")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if use_full_system:
                # Full pipeline
                annotated = system.process_frame(frame)
            else:
                # Demo mode: just show the frame with basic overlay
                annotated = frame.copy()
                cv2.putText(annotated, f"Frame: {frame_count}",
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2)
                cv2.putText(annotated,
                           f"Dashboard: http://localhost:{args.port}",
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (200, 200, 200), 1)

            # Push frame to dashboard
            server.update_frame(annotated)

            # Show locally too
            cv2.imshow("Attention System", annotated)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    except KeyboardInterrupt:
        print("\n  Shutting down...")

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()
    summary = scoring.get_class_summary()
    logger.end_session(
        total_frames=frame_count,
        avg_score=summary.get("avg_score", 0),
        total_students=summary.get("total", 0),
    )
    server.stop()
    db.close()
    print("  ✅ Shutdown complete")


if __name__ == "__main__":
    main()

