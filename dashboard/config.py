

"""
dashboard/config.py
Dashboard configuration.
"""
from dataclasses import dataclass


@dataclass
class DashboardConfig:
    """Web dashboard configuration."""

    # ── Server ──
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # ── WebSocket ──
    ws_update_interval_ms: int = 500     # Push updates every 500ms
    ws_max_clients: int = 10

    # ── MJPEG Stream ──
    stream_quality: int = 70             # JPEG quality (0-100)
    stream_fps: int = 15                 # Max FPS for web stream
    stream_width: int = 960              # Resize for streaming

    # ── Dashboard ──
    chart_history_seconds: int = 300     # 5 min of chart history
    leaderboard_size: int = 10
    alert_display_count: int = 20

    # ── Export ──
    export_folder: str = "exports"


dashboard_cfg = DashboardConfig()
