

"""
dashboard/server.py
Launches the FastAPI dashboard server in a background thread
so it runs alongside the main CV processing loop.
"""
import threading
import time
import cv2
import numpy as np
from typing import Optional, Callable

import uvicorn

from .config import dashboard_cfg
from .api import create_app


class DashboardServer:
    """
    Manages the web dashboard server lifecycle.

    Runs FastAPI + uvicorn in a background thread so the main
    OpenCV loop continues uninterrupted.

    Usage:
        from dashboard import DashboardServer
        from attention_scoring import ScoringPipeline
        from database import DatabaseManager, LiveLogger, AnalyticsEngine

        scoring = ScoringPipeline()
        db = DatabaseManager()
        logger = LiveLogger(db)
        analytics = AnalyticsEngine(db)

        server = DashboardServer(
            scoring=scoring,
            db=db,
            analytics=analytics,
            logger=logger,
        )
        server.start(port=8000)

        # In your frame loop:
        annotated_frame = process(frame)
        server.update_frame(annotated_frame)

        # On shutdown:
        server.stop()
    """

    def __init__(self, scoring=None, db=None, analytics=None, logger=None):
        self._scoring = scoring
        self._db = db
        self._analytics = analytics
        self._logger = logger

        self._latest_frame: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self._server_thread: Optional[threading.Thread] = None
        self._uvicorn_server = None
        self._running = False

    def _get_frame(self) -> Optional[bytes]:
        """Called by the API to get latest JPEG frame."""
        with self._frame_lock:
            return self._latest_frame

    def update_frame(self, frame: np.ndarray):
        """
        Update the latest frame for MJPEG streaming.

        Call this from your main processing loop after annotating the frame.

        Args:
            frame: BGR numpy array (annotated frame from OpenCV)
        """
        # Resize for streaming
        h, w = frame.shape[:2]
        target_w = dashboard_cfg.stream_width
        if w > target_w:
            scale = target_w / w
            frame = cv2.resize(frame, (target_w, int(h * scale)))

        # Encode as JPEG
        _, buffer = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, dashboard_cfg.stream_quality]
        )

        with self._frame_lock:
            self._latest_frame = buffer.tobytes()

    def start(self, host: Optional[str] = None, port: Optional[int] = None):
        """
        Start the dashboard server in a background thread.

        Args:
            host: Bind address (default: 0.0.0.0)
            port: Port number (default: 8000)
        """
        host = host or dashboard_cfg.host
        port = port or dashboard_cfg.port

        # Create FastAPI app
        app = create_app(
            scoring=self._scoring,
            db=self._db,
            analytics=self._analytics,
            logger=self._logger,
            frame_source=self._get_frame,
        )

        # Configure uvicorn
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        self._uvicorn_server = uvicorn.Server(config)

        # Run in background thread
        self._server_thread = threading.Thread(
            target=self._uvicorn_server.run,
            daemon=True,
            name="DashboardServer",
        )
        self._server_thread.start()
        self._running = True

        # Wait a moment for startup
        time.sleep(0.5)
        print(f"  🌐 Dashboard running at: http://localhost:{port}")
        print(f"     Video feed: http://localhost:{port}/video_feed")
        print(f"     API docs:   http://localhost:{port}/docs")

    def stop(self):
        """Stop the dashboard server."""
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
        if self._server_thread:
            self._server_thread.join(timeout=3)
        self._running = False
        print("  🌐 Dashboard server stopped.")

    @property
    def is_running(self) -> bool:
        return self._running