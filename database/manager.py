
"""
database/manager.py
Central database connection manager with auto-schema creation,
WAL mode, connection pooling, and migration support.
"""
import sqlite3
import threading
import os
from typing import Optional, List, Tuple, Any
from contextlib import contextmanager
from .config import db_cfg
from .schema import SCHEMA_SQL


class DatabaseManager:
    """
    Central SQLite database manager.

    Features:
        - Auto-creates tables on first run
        - WAL mode for concurrent read/write performance
        - Thread-safe with per-thread connections
        - Context manager for transactions
        - Migration support via schema_version table

    Usage:
        db = DatabaseManager()                          # Uses default path
        db = DatabaseManager("custom_path.db")          # Custom path

        with db.connection() as conn:
            conn.execute("INSERT INTO ...", (...))

        rows = db.fetch_all("SELECT * FROM sessions")
        db.close()
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or db_cfg.db_path
        self._local = threading.local()
        self._lock = threading.Lock()

        # Bootstrap
        self._init_database()

    # ──────────── Connection Management ────────────

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                self._db_path,
                timeout=db_cfg.busy_timeout_ms / 1000.0,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row  # Dict-like rows
            conn.execute("PRAGMA foreign_keys = ON")
            if db_cfg.wal_mode:
                conn.execute(f"PRAGMA journal_mode = {db_cfg.journal_mode}")
            conn.execute("PRAGMA synchronous = NORMAL")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def connection(self):
        """
        Context manager for database transactions.

        Usage:
            with db.connection() as conn:
                conn.execute("INSERT INTO ...", (...))
                # Auto-commits on exit, auto-rollbacks on exception
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ──────────── Query Helpers ────────────

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a single SQL statement and commit."""
        with self.connection() as conn:
            cursor = conn.execute(sql, params)
            return cursor

    def execute_many(self, sql: str, param_list: List[tuple]):
        """Execute a SQL statement with many parameter sets (batch insert)."""
        with self.connection() as conn:
            conn.executemany(sql, param_list)

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Execute and fetch a single row."""
        conn = self._get_connection()
        return conn.execute(sql, params).fetchone()

    def fetch_all(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute and fetch all rows."""
        conn = self._get_connection()
        return conn.execute(sql, params).fetchall()

    def fetch_value(self, sql: str, params: tuple = ()) -> Any:
        """Execute and fetch a single scalar value."""
        row = self.fetch_one(sql, params)
        return row[0] if row else None

    # ──────────── Schema Management ────────────

    def _init_database(self):
        """Create tables and indexes if they don't exist."""
        with self.connection() as conn:
            conn.executescript(SCHEMA_SQL)

            # Record schema version
            existing = conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()[0]

            if existing is None or existing < db_cfg.schema_version:
                conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                    (db_cfg.schema_version,)
                )

        if db_cfg.vacuum_on_startup:
            self._get_connection().execute("VACUUM")

    def get_schema_version(self) -> int:
        """Get current schema version."""
        return self.fetch_value("SELECT MAX(version) FROM schema_version") or 0

    # ──────────── Maintenance ────────────

    def cleanup_old_data(self, days: Optional[int] = None):
        """Delete data older than N days."""
        days = days or db_cfg.retention_days
        self.execute(
            "DELETE FROM attention_scores WHERE created_at < datetime('now', ?)",
            (f'-{days} days',)
        )
        self.execute(
            "DELETE FROM alerts WHERE created_at < datetime('now', ?)",
            (f'-{days} days',)
        )
        self.execute(
            "DELETE FROM sessions WHERE created_at < datetime('now', ?) "
            "AND status = 'completed'",
            (f'-{days} days',)
        )

    def vacuum(self):
        """Reclaim disk space after deletions."""
        self._get_connection().execute("VACUUM")

    def get_db_size_mb(self) -> float:
        """Get database file size in MB."""
        if os.path.exists(self._db_path):
            return os.path.getsize(self._db_path) / (1024 * 1024)
        return 0.0

    def get_table_counts(self) -> dict:
        """Get row counts for all tables."""
        tables = ["sessions", "students", "attention_scores",
                  "alerts", "session_summaries"]
        counts = {}
        for t in tables:
            counts[t] = self.fetch_value(f"SELECT COUNT(*) FROM {t}") or 0
        return counts

    def close(self):
        """Close the thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

