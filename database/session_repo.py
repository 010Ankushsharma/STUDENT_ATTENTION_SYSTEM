
"""
database/session_repo.py
CRUD operations for monitoring sessions.
"""
import uuid
import json
from datetime import datetime
from typing import Optional, List, Dict
from .manager import DatabaseManager


class SessionRepository:
    """
    Manages monitoring session records.

    Usage:
        repo = SessionRepository(db)
        session_id = repo.create("Morning Class", camera_index=0)
        repo.end_session(session_id, total_frames=5400)
        sessions = repo.list_recent(limit=10)
    """

    def __init__(self, db: DatabaseManager):
        self._db = db

    def create(self,
               name: str = "Unnamed Session",
               camera_index: int = 0,
               config: Optional[Dict] = None,
               notes: str = "") -> str:
        """
        Create a new monitoring session.

        Args:
            name:          Human-readable session name.
            camera_index:  Camera used.
            config:        Optional config dict (serialised as JSON).
            notes:         Optional notes.

        Returns:
            session_id (UUID string)
        """
        session_id = str(uuid.uuid4())[:12]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config_json = json.dumps(config) if config else None

        self._db.execute(
            """INSERT INTO sessions
               (session_id, name, start_time, camera_index, status,
                config_json, notes, created_at)
               VALUES (?, ?, ?, ?, 'active', ?, ?, ?)""",
            (session_id, name, now, camera_index, config_json, notes, now)
        )
        return session_id

    def end_session(self, session_id: str,
                    total_frames: int = 0,
                    avg_score: float = 0.0,
                    total_students: int = 0):
        """Mark a session as completed."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._db.execute(
            """UPDATE sessions
               SET end_time = ?, status = 'completed',
                   total_frames = ?, avg_class_score = ?,
                   total_students = ?
               WHERE session_id = ?""",
            (now, total_frames, avg_score, total_students, session_id)
        )

    def get(self, session_id: str) -> Optional[dict]:
        """Get session by ID."""
        row = self._db.fetch_one(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        return dict(row) if row else None

    def list_recent(self, limit: int = 20) -> List[dict]:
        """List recent sessions, newest first."""
        rows = self._db.fetch_all(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        return [dict(r) for r in rows]

    def list_active(self) -> List[dict]:
        """List currently active sessions."""
        rows = self._db.fetch_all(
            "SELECT * FROM sessions WHERE status = 'active' "
            "ORDER BY start_time DESC"
        )
        return [dict(r) for r in rows]

    def delete(self, session_id: str):
        """Delete a session and all related data (CASCADE)."""
        self._db.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,)
        )

    def update_notes(self, session_id: str, notes: str):
        """Update session notes."""
        self._db.execute(
            "UPDATE sessions SET notes = ? WHERE session_id = ?",
            (notes, session_id)
        )

    def count(self) -> int:
        """Total number of sessions."""
        return self._db.fetch_value("SELECT COUNT(*) FROM sessions") or 0

