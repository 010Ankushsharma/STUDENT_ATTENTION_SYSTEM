"""
database/alert_repo.py
CRUD operations for alert events.
"""
from datetime import datetime
from typing import List, Optional
from .manager import DatabaseManager


class AlertRepository:
    """
    Store and query alert events.

    Usage:
        repo = AlertRepository(db)
        repo.insert(session_id, student_id=2, alert_type="sleepy",
                    severity="critical", message="Student 2 sleeping!")
        alerts = repo.get_session_alerts(session_id)
    """

    def __init__(self, db: DatabaseManager):
        self._db = db

    def insert(self,
               session_id: str,
               student_id: int,
               alert_type: str,
               severity: str,
               message: str,
               score: float = 0.0,
               state: str = "",
               sustained_frames: int = 0):
        """Insert a new alert record."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._db.execute(
            """INSERT INTO alerts
               (session_id, student_id, alert_type, severity,
                message, score, state, sustained_frames, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, student_id, alert_type, severity,
             message, score, state, sustained_frames, now)
        )

    def get_session_alerts(self, session_id: str) -> List[dict]:
        """Get all alerts for a session."""
        rows = self._db.fetch_all(
            "SELECT * FROM alerts WHERE session_id = ? ORDER BY timestamp DESC",
            (session_id,)
        )
        return [dict(r) for r in rows]

    def get_student_alerts(self, session_id: str, student_id: int) -> List[dict]:
        """Get alerts for a specific student in a session."""
        rows = self._db.fetch_all(
            """SELECT * FROM alerts
               WHERE session_id = ? AND student_id = ?
               ORDER BY timestamp DESC""",
            (session_id, student_id)
        )
        return [dict(r) for r in rows]

    def get_by_severity(self, session_id: str, severity: str) -> List[dict]:
        """Get alerts filtered by severity."""
        rows = self._db.fetch_all(
            """SELECT * FROM alerts
               WHERE session_id = ? AND severity = ?
               ORDER BY timestamp DESC""",
            (session_id, severity)
        )
        return [dict(r) for r in rows]

    def acknowledge(self, alert_id: int):
        """Mark an alert as acknowledged."""
        self._db.execute(
            "UPDATE alerts SET acknowledged = 1 WHERE id = ?",
            (alert_id,)
        )

    def count(self, session_id: Optional[str] = None) -> int:
        if session_id:
            return self._db.fetch_value(
                "SELECT COUNT(*) FROM alerts WHERE session_id = ?",
                (session_id,)
            ) or 0
        return self._db.fetch_value("SELECT COUNT(*) FROM alerts") or 0

