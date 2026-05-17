
"""
database/student_repo.py
CRUD operations for student profiles.
"""
from datetime import datetime
from typing import Optional, List
from .manager import DatabaseManager


class StudentRepository:
    """
    Manages persistent student records.

    Usage:
        repo = StudentRepository(db)
        repo.upsert(track_id=3, label="Row 2, Seat 5")
        student = repo.get_by_track_id(3)
        all_students = repo.list_all()
    """

    def __init__(self, db: DatabaseManager):
        self._db = db

    def upsert(self, track_id: int, label: str = "") -> int:
        """
        Insert or update a student by track ID.

        Returns:
            student_db_id (primary key)
        """
        existing = self._db.fetch_one(
            "SELECT student_db_id FROM students WHERE student_track_id = ?",
            (track_id,)
        )

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if existing:
            db_id = existing["student_db_id"]
            self._db.execute(
                "UPDATE students SET updated_at = ?, label = ? "
                "WHERE student_db_id = ?",
                (now, label, db_id) if label else (now, existing.get("label", ""), db_id)
            )
            return db_id
        else:
            cursor = self._db.execute(
                """INSERT INTO students
                   (student_track_id, label, first_seen, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (track_id, label, now, now, now)
            )
            return cursor.lastrowid

    def get_by_track_id(self, track_id: int) -> Optional[dict]:
        """Get student by tracker-assigned ID."""
        row = self._db.fetch_one(
            "SELECT * FROM students WHERE student_track_id = ?",
            (track_id,)
        )
        return dict(row) if row else None

    def list_all(self) -> List[dict]:
        """List all students."""
        rows = self._db.fetch_all(
            "SELECT * FROM students ORDER BY student_track_id"
        )
        return [dict(r) for r in rows]

    def update_stats(self, track_id: int, sessions: int, avg_attention: float):
        """Update aggregated stats for a student."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._db.execute(
            """UPDATE students
               SET total_sessions = ?, avg_attention_pct = ?, updated_at = ?
               WHERE student_track_id = ?""",
            (sessions, avg_attention, now, track_id)
        )

    def set_label(self, track_id: int, label: str):
        """Assign a human-readable label to a student."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._db.execute(
            "UPDATE students SET label = ?, updated_at = ? "
            "WHERE student_track_id = ?",
            (label, now, track_id)
        )

    def delete(self, track_id: int):
        """Remove a student profile."""
        self._db.execute(
            "DELETE FROM students WHERE student_track_id = ?",
            (track_id,)
        )

    def count(self) -> int:
        return self._db.fetch_value("SELECT COUNT(*) FROM students") or 0