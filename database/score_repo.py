
"""
database/score_repo.py
CRUD operations for attention score records.
"""
from datetime import datetime
from typing import List, Optional, Dict
from .manager import DatabaseManager


class ScoreRepository:
    """
    Insert and query per-frame attention score records.

    Usage:
        repo = ScoreRepository(db)
        repo.insert(session_id, student_id=0, frame_num=42,
                    score=0.85, state="attentive", ...)
        history = repo.get_student_history(session_id, student_id=0)
        timeline = repo.get_timeline(session_id)
    """

    def __init__(self, db: DatabaseManager):
        self._db = db

    def insert(self,
               session_id: str,
               student_id: int,
               frame_num: int,
               score: float,
               state: str,
               ear: float = 0.0,
               blink_rate: float = 0.0,
               perclos: float = 0.0,
               gaze_direction: str = "center",
               drowsiness: str = "alert",
               yaw: float = 0.0,
               pitch: float = 0.0,
               roll: float = 0.0,
               head_direction: str = "forward",
               hp_score: float = 0.0,
               gaze_score: float = 0.0,
               ear_score: float = 0.0,
               blink_score: float = 0.0,
               perclos_score: float = 0.0,
               ):
        """Insert a single attention score record."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._db.execute(
            """INSERT INTO attention_scores
               (session_id, student_id, frame_num, timestamp,
                score, state, ear, blink_rate, perclos,
                gaze_direction, drowsiness, yaw, pitch, roll,
                head_direction, hp_score, gaze_score,
                ear_score, blink_score, perclos_score)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (session_id, student_id, frame_num, now,
             score, state, ear, blink_rate, perclos,
             gaze_direction, drowsiness, yaw, pitch, roll,
             head_direction, hp_score, gaze_score,
             ear_score, blink_score, perclos_score)
        )

    def insert_batch(self, records: List[tuple]):
        """
        Batch insert score records for performance.

        Each tuple must match the INSERT column order above.
        """
        self._db.execute_many(
            """INSERT INTO attention_scores
               (session_id, student_id, frame_num, timestamp,
                score, state, ear, blink_rate, perclos,
                gaze_direction, drowsiness, yaw, pitch, roll,
                head_direction, hp_score, gaze_score,
                ear_score, blink_score, perclos_score)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            records
        )

    def get_student_history(self, session_id: str, student_id: int,
                            limit: int = 1000) -> List[dict]:
        """Get score history for one student in a session."""
        rows = self._db.fetch_all(
            """SELECT * FROM attention_scores
               WHERE session_id = ? AND student_id = ?
               ORDER BY frame_num DESC LIMIT ?""",
            (session_id, student_id, limit)
        )
        return [dict(r) for r in rows]

    def get_timeline(self, session_id: str, interval_sec: int = 10) -> List[dict]:
        """
        Get averaged scores at regular intervals for timeline charts.

        Groups by N-second intervals and averages the score.
        """
        rows = self._db.fetch_all(
            """SELECT
                 student_id,
                 (frame_num / (? * 30)) as time_bucket,
                 ROUND(AVG(score), 3) as avg_score,
                 MIN(score) as min_score,
                 MAX(score) as max_score,
                 COUNT(*) as samples,
                 -- Most common state in bucket
                 state
               FROM attention_scores
               WHERE session_id = ?
               GROUP BY student_id, time_bucket
               ORDER BY student_id, time_bucket""",
            (interval_sec, session_id)
        )
        return [dict(r) for r in rows]

    def get_latest_per_student(self, session_id: str) -> List[dict]:
        """Get the most recent score for each student."""
        rows = self._db.fetch_all(
            """SELECT a.* FROM attention_scores a
               INNER JOIN (
                   SELECT student_id, MAX(frame_num) as max_frame
                   FROM attention_scores
                   WHERE session_id = ?
                   GROUP BY student_id
               ) b ON a.student_id = b.student_id
                   AND a.frame_num = b.max_frame
                   AND a.session_id = ?""",
            (session_id, session_id)
        )
        return [dict(r) for r in rows]

    def count_by_state(self, session_id: str) -> Dict[str, int]:
        """Count records per state for a session."""
        rows = self._db.fetch_all(
            """SELECT state, COUNT(*) as cnt
               FROM attention_scores
               WHERE session_id = ?
               GROUP BY state""",
            (session_id,)
        )
        return {r["state"]: r["cnt"] for r in rows}

    def count(self, session_id: Optional[str] = None) -> int:
        """Count total score records, optionally filtered by session."""
        if session_id:
            return self._db.fetch_value(
                "SELECT COUNT(*) FROM attention_scores WHERE session_id = ?",
                (session_id,)
            ) or 0
        return self._db.fetch_value(
            "SELECT COUNT(*) FROM attention_scores"
        ) or 0

