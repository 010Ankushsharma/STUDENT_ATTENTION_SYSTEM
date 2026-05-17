
"""
main_config.py
Global configuration for the Student Attention Detection System.
Imports and orchestrates settings from all sub-modules.
"""
from dataclasses import dataclass


@dataclass
class SystemConfig:
    """Top-level system configuration."""

    # ── Camera ──
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    target_fps: int = 30

    # ── Processing ──
    process_width: int = 640          # Resize for speed
    skip_frames: int = 0             # 0 = process every frame

    # ── Display ──
    show_video: bool = True
    window_name: str = "Student Attention Detection System"
    show_fps: bool = True
    show_dashboard: bool = True       # On-screen metrics panel

    # ── Logging ──
    log_to_console: bool = True
    log_to_file: bool = False
    log_file: str = "attention_log.csv"
    log_interval_sec: float = 5.0     # Log every N seconds

    # ── Alerts ──
    alert_sound: bool = False         # Play sound on alert
    alert_visual: bool = True         # Flash screen border on alert

    # ── Session ──
    session_name: str = "classroom_session"
    auto_screenshot: bool = False
    screenshot_interval_sec: float = 60.0

    # ── Web Dashboard ──
    dashboard_enabled: bool = True
    dashboard_port: int = 8000
    dashboard_auto_open: bool = False     # Auto-open browser on start


sys_cfg = SystemConfig()