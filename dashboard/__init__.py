
"""
Dashboard Module
=================
Web-based real-time dashboard for the Student Attention Detection System.

Tech Stack:
    - Backend:  FastAPI + WebSockets
    - Frontend: HTML5 / CSS3 / Vanilla JS (no frameworks needed)
    - Charts:   Chart.js (CDN)
    - Updates:  WebSocket for real-time push

Features:
    - Live webcam feed (MJPEG stream)
    - Real-time attention scores per student
    - Class-wide attention charts
    - Alert notifications panel
    - Session analytics & history
    - Export to CSV/Excel
    - Responsive design (desktop + tablet)

Usage:
    from dashboard import DashboardServer
    server = DashboardServer(scoring_pipeline, db_manager)
    server.start(port=8000)

    # Then open: http://localhost:8000
"""
from .server import DashboardServer
from .api import create_app

__all__ = ["DashboardServer", "create_app"]
__version__ = "1.0.0"
