
"""
database/config.py
All database configuration parameters.
"""
from dataclasses import dataclass


@dataclass
class DBConfig:
    """Database configuration."""

    # ── Connection ──
    db_path: str = "attention_system.db"
    wal_mode: bool = True  # Write-Ahead Logging (faster writes)
    journal_mode: str = "WAL"
    busy_timeout_ms: int = 5000

    # ── Live Logger ──
    batch_size: int = 50  # Flush every N records
    flush_interval_sec: float = 3.0  # OR flush every N seconds
    log_every_n_frames: int = 5  # Only log every Nth frame per student

    # ── Retention ──
    retention_days: int = 90  # Auto-delete data older than N days
    vacuum_on_startup: bool = False

    # ── Schema version (for future migrations) ──
    schema_version: int = 1


db_cfg = DBConfig()
