"""
core/config_loader.py
Load configuration from environment variables and .env files.

Usage:
    from core import load_config_from_env
    load_config_from_env()  # Applies env vars to main_config
"""
import os
from typing import Optional


def load_config_from_env():
    """
    Override main_config settings from environment variables.

    Environment variables (all optional):
        CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
        PROCESS_WIDTH, TARGET_FPS, SKIP_FRAMES
        DASHBOARD_ENABLED, DASHBOARD_PORT, DASHBOARD_HOST
        DB_PATH, DB_WAL_MODE, DB_RETENTION_DAYS
        LOG_LEVEL, LOG_TO_FILE, LOG_DIR
        ALERT_SOUND, ALERT_VISUAL
        BATCH_SIZE, FLUSH_INTERVAL, LOG_EVERY_N_FRAMES
    """
    from main_config import sys_cfg
    from database.config import db_cfg
    from dashboard.config import dashboard_cfg

    # ── Camera ──
    if v := os.getenv("CAMERA_INDEX"):
        sys_cfg.camera_index = int(v)
    if v := os.getenv("CAMERA_WIDTH"):
        sys_cfg.camera_width = int(v)
    if v := os.getenv("CAMERA_HEIGHT"):
        sys_cfg.camera_height = int(v)

    # ── Processing ──
    if v := os.getenv("PROCESS_WIDTH"):
        sys_cfg.process_width = int(v)
    if v := os.getenv("TARGET_FPS"):
        sys_cfg.target_fps = int(v)
    if v := os.getenv("SKIP_FRAMES"):
        sys_cfg.skip_frames = int(v)

    # ── Dashboard ──
    if v := os.getenv("DASHBOARD_ENABLED"):
        sys_cfg.dashboard_enabled = v.lower() in ("true", "1", "yes")
    if v := os.getenv("DASHBOARD_PORT"):
        sys_cfg.dashboard_port = int(v)
        dashboard_cfg.port = int(v)
    if v := os.getenv("DASHBOARD_HOST"):
        dashboard_cfg.host = v

    # ── Database ──
    if v := os.getenv("DB_PATH"):
        db_cfg.db_path = v
    if v := os.getenv("ATTENTION_DB_PATH"):
        db_cfg.db_path = v
    if v := os.getenv("DB_WAL_MODE"):
        db_cfg.wal_mode = v.lower() in ("true", "1", "yes")
    if v := os.getenv("DB_RETENTION_DAYS"):
        db_cfg.retention_days = int(v)

    # ── Logging ──
    if v := os.getenv("LOG_TO_FILE"):
        sys_cfg.log_to_file = v.lower() in ("true", "1", "yes")

    # ── Alerts ──
    if v := os.getenv("ALERT_SOUND"):
        sys_cfg.alert_sound = v.lower() in ("true", "1", "yes")
    if v := os.getenv("ALERT_VISUAL"):
        sys_cfg.alert_visual = v.lower() in ("true", "1", "yes")

    # ── Performance ──
    if v := os.getenv("BATCH_SIZE"):
        db_cfg.batch_size = int(v)
    if v := os.getenv("FLUSH_INTERVAL"):
        db_cfg.flush_interval_sec = float(v)
    if v := os.getenv("LOG_EVERY_N_FRAMES"):
        db_cfg.log_every_n_frames = int(v)


def _load_dotenv():
    """Load .env file if it exists (without python-dotenv dependency)."""
    env_file = ".env"
    if not os.path.exists(env_file):
        return

    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


# Auto-load .env on import
_load_dotenv()

