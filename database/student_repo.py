"""
database/student_repo.py
CRUD operations for student profiles.
"""
from datetime import datetime
from typing import Optional, List
from .manager import DatabaseManager


class StudentRepository:
    def __init__(self, db: DatabaseManager):
        self._db = db

    def upsert(self, track_id: int, label: str = "") -> int:
        existing = self._db.fetch_one(
            "SELECT student_db_id, label FROM students WHERE student_track_id = ?",
            (track_id,)
        )
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if existing:
            db_id = existing[0]
            existing_label = existing[1] or ""
            self._db.execute(
                "UPDATE students SET updated_at = ?, label = ? WHERE student_db_id = ?",
                (now, label if label else existing_label, db_id)
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
        row = self._db.fetch_one(
            "SELECT * FROM students WHERE student_track_id = ?",
            (track_id,)
        )
        return dict(row) if row else None

    def list_all(self) -> List[dict]:
        rows = self._db.fetch_all(
            "SELECT * FROM students ORDER BY student_track_id"
        )
        return [dict(r) for r in rows]

    def update_stats(self, track_id: int, sessions: int, avg_attention: float):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._db.execute(
            """UPDATE students
               SET total_sessions = ?, avg_attention_pct = ?, updated_at = ?
               WHERE student_track_id = ?""",
            (sessions, avg_attention, now, track_id)
        )

    def set_label(self, track_id: int, label: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._db.execute(
            "UPDATE students SET label = ?, updated_at = ? WHERE student_track_id = ?",
            (label, now, track_id)
        )

    def delete(self, track_id: int):
        self._db.execute(
            "DELETE FROM students WHERE student_track_id = ?",
            (track_id,)
        )

    def count(self) -> int:
        return self._db.fetch_value("SELECT COUNT(*) FROM students") or 0
