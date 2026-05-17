
"""
database/analytics.py
Pre-built analytical queries for reports and dashboards.

Provides:
    - Per-student session summaries
    - Class-wide statistics
    - Attention timelines for charting
    - State distribution analysis
    - Trend analysis across sessions
    - Peak / low attention periods
    - Student ranking & comparison
"""
from typing import List, Dict, Optional
from .manager import DatabaseManager


class AnalyticsEngine:
    """
    Pre-built analytics queries for attention data.

    Usage:
        analytics = AnalyticsEngine(db)

        # Session-level
        summary = analytics.session_overview("abc123")
        timeline = analytics.attention_timeline("abc123")

        # Student-level
        report = analytics.student_summary("abc123", student_id=0)
        trend = analytics.student_trend_across_sessions(student_id=0)

        # Class-level
        ranking = analytics.class_ranking("abc123")
        distribution = analytics.state_distribution("abc123")
    """

    def __init__(self, db: DatabaseManager):
        self._db = db

    # ──────────── Session-Level Analytics ────────────

    def session_overview(self, session_id: str) -> dict:
        """
        Complete overview of a monitoring session.

        Returns:
            {
                "session": {...},
                "total_students": 15,
                "total_frames": 5400,
                "avg_score": 0.72,
                "state_counts": {"attentive": 3200, ...},
                "alert_count": 5,
                "duration_minutes": 30.0,
            }
        """
        session = self._db.fetch_one(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        if not session:
            return {"error": "Session not found"}

        total_frames = self._db.fetch_value(
            "SELECT COUNT(*) FROM attention_scores WHERE session_id = ?",
            (session_id,)
        ) or 0

        avg_score = self._db.fetch_value(
            "SELECT ROUND(AVG(score), 3) FROM attention_scores WHERE session_id = ?",
            (session_id,)
        ) or 0.0

        students = self._db.fetch_value(
            "SELECT COUNT(DISTINCT student_id) FROM attention_scores WHERE session_id = ?",
            (session_id,)
        ) or 0

        state_rows = self._db.fetch_all(
            """SELECT state, COUNT(*) as cnt
               FROM attention_scores WHERE session_id = ?
               GROUP BY state""",
            (session_id,)
        )
        state_counts = {r["state"]: r["cnt"] for r in state_rows}

        alert_count = self._db.fetch_value(
            "SELECT COUNT(*) FROM alerts WHERE session_id = ?",
            (session_id,)
        ) or 0

        return {
            "session": dict(session),
            "total_students": students,
            "total_score_records": total_frames,
            "avg_score": avg_score,
            "state_counts": state_counts,
            "alert_count": alert_count,
        }

    def attention_timeline(self, session_id: str,
                           bucket_seconds: int = 10) -> List[dict]:
        """
        Attention scores averaged over time buckets for charting.

        Returns list of:
            {"student_id": 0, "time_bucket": 5, "avg_score": 0.82, ...}
        """
        rows = self._db.fetch_all(
            """SELECT
                 student_id,
                 CAST(frame_num / (? * 30) AS INTEGER) AS time_bucket,
                 ROUND(AVG(score), 3) AS avg_score,
                 ROUND(MIN(score), 3) AS min_score,
                 ROUND(MAX(score), 3) AS max_score,
                 COUNT(*) AS samples
               FROM attention_scores
               WHERE session_id = ?
               GROUP BY student_id, time_bucket
               ORDER BY student_id, time_bucket""",
            (bucket_seconds, session_id)
        )
        return [dict(r) for r in rows]

    def class_score_over_time(self, session_id: str,
                              bucket_seconds: int = 10) -> List[dict]:
        """
        Class-average score over time (all students combined).

        Returns list of:
            {"time_bucket": 0, "avg_score": 0.78, "num_students": 12}
        """
        rows = self._db.fetch_all(
            """SELECT
                 CAST(frame_num / (? * 30) AS INTEGER) AS time_bucket,
                 ROUND(AVG(score), 3) AS avg_score,
                 COUNT(DISTINCT student_id) AS num_students,
                 COUNT(*) AS samples
               FROM attention_scores
               WHERE session_id = ?
               GROUP BY time_bucket
               ORDER BY time_bucket""",
            (bucket_seconds, session_id)
        )
        return [dict(r) for r in rows]

    # ──────────── Student-Level Analytics ────────────

    def student_summary(self, session_id: str, student_id: int) -> dict:
        """
        Detailed summary for one student in one session.

        Returns:
            {
                "student_id": 0,
                "avg_score": 0.82,
                "min_score": 0.35,
                "max_score": 0.98,
                "state_distribution": {"attentive": 72.5, ...},
                "avg_ear": 0.28,
                "avg_blink_rate": 16.2,
                "total_frames": 450,
                "alert_count": 1,
            }
        """
        row = self._db.fetch_one(
            """SELECT
                 COUNT(*) AS total_frames,
                 ROUND(AVG(score), 3) AS avg_score,
                 ROUND(MIN(score), 3) AS min_score,
                 ROUND(MAX(score), 3) AS max_score,
                 ROUND(AVG(ear), 4) AS avg_ear,
                 ROUND(AVG(blink_rate), 1) AS avg_blink_rate,
                 ROUND(AVG(perclos), 4) AS avg_perclos,
                 ROUND(AVG(ABS(yaw)), 1) AS avg_abs_yaw,
                 ROUND(AVG(ABS(pitch)), 1) AS avg_abs_pitch
               FROM attention_scores
               WHERE session_id = ? AND student_id = ?""",
            (session_id, student_id)
        )

        state_rows = self._db.fetch_all(
            """SELECT state, COUNT(*) AS cnt
               FROM attention_scores
               WHERE session_id = ? AND student_id = ?
               GROUP BY state""",
            (session_id, student_id)
        )

        total = row["total_frames"] if row else 1
        state_dist = {}
        for sr in state_rows:
            state_dist[sr["state"]] = round(sr["cnt"] / total * 100, 1)

        alert_count = self._db.fetch_value(
            "SELECT COUNT(*) FROM alerts WHERE session_id = ? AND student_id = ?",
            (session_id, student_id)
        ) or 0

        return {
            "student_id": student_id,
            "session_id": session_id,
            "total_frames": total,
            "avg_score": row["avg_score"] if row else 0,
            "min_score": row["min_score"] if row else 0,
            "max_score": row["max_score"] if row else 0,
            "state_distribution": state_dist,
            "avg_ear": row["avg_ear"] if row else 0,
            "avg_blink_rate": row["avg_blink_rate"] if row else 0,
            "avg_perclos": row["avg_perclos"] if row else 0,
            "avg_abs_yaw": row["avg_abs_yaw"] if row else 0,
            "avg_abs_pitch": row["avg_abs_pitch"] if row else 0,
            "alert_count": alert_count,
        }

    def student_trend_across_sessions(self, student_id: int,
                                      limit: int = 20) -> List[dict]:
        """
        Track a student's attention trend across multiple sessions.

        Returns list of:
            {"session_id": "abc", "avg_score": 0.78, "state_dist": {...}, ...}
        """
        rows = self._db.fetch_all(
            """SELECT
                 session_id,
                 ROUND(AVG(score), 3) AS avg_score,
                 COUNT(*) AS total_frames,
                 MIN(timestamp) AS start_time
               FROM attention_scores
               WHERE student_id = ?
               GROUP BY session_id
               ORDER BY start_time DESC
               LIMIT ?""",
            (student_id, limit)
        )
        return [dict(r) for r in rows]

    # ──────────── Class-Level Analytics ────────────

    def class_ranking(self, session_id: str) -> List[dict]:
        """
        Rank students by attention score in a session.

        Returns:
            [{"rank": 1, "student_id": 3, "avg_score": 0.92, ...}, ...]
        """
        rows = self._db.fetch_all(
            """SELECT
                 student_id,
                 ROUND(AVG(score), 3) AS avg_score,
                 COUNT(*) AS total_frames,
                 ROUND(
                   SUM(CASE WHEN state='attentive' THEN 1.0 ELSE 0.0 END)
                   / COUNT(*) * 100, 1
                 ) AS attentive_pct
               FROM attention_scores
               WHERE session_id = ?
               GROUP BY student_id
               ORDER BY avg_score DESC""",
            (session_id,)
        )
        return [
            {**dict(r), "rank": i + 1}
            for i, r in enumerate(rows)
        ]

    def state_distribution(self, session_id: str) -> dict:
        """
        Overall state distribution for entire class.

        Returns:
            {"attentive": 65.2, "distracted": 22.1, "sleepy": 5.3, ...}
        """
        total = self._db.fetch_value(
            "SELECT COUNT(*) FROM attention_scores WHERE session_id = ?",
            (session_id,)
        ) or 1

        rows = self._db.fetch_all(
            """SELECT state, COUNT(*) AS cnt
               FROM attention_scores WHERE session_id = ?
               GROUP BY state""",
            (session_id,)
        )
        return {
            r["state"]: round(r["cnt"] / total * 100, 1)
            for r in rows
        }

    def peak_attention_periods(self, session_id: str,
                               bucket_seconds: int = 30) -> dict:
        """
        Find the best and worst attention periods.

        Returns:
            {"best": {"time_bucket": 5, "avg_score": 0.92},
             "worst": {"time_bucket": 12, "avg_score": 0.35}}
        """
        rows = self._db.fetch_all(
            """SELECT
                 CAST(frame_num / (? * 30) AS INTEGER) AS time_bucket,
                 ROUND(AVG(score), 3) AS avg_score
               FROM attention_scores
               WHERE session_id = ?
               GROUP BY time_bucket
               ORDER BY avg_score DESC""",
            (bucket_seconds, session_id)
        )
        if not rows:
            return {"best": None, "worst": None}
        return {
            "best": dict(rows[0]),
            "worst": dict(rows[-1]),
        }

    def students_needing_attention(self, session_id: str,
                                   threshold: float = 0.5) -> List[dict]:
        """
        Find students with avg score below threshold.

        Returns list sorted by score (worst first).
        """
        rows = self._db.fetch_all(
            """SELECT
                 student_id,
                 ROUND(AVG(score), 3) AS avg_score,
                 COUNT(*) AS total_frames
               FROM attention_scores
               WHERE session_id = ?
               GROUP BY student_id
               HAVING AVG(score) < ?
               ORDER BY avg_score ASC""",
            (session_id, threshold)
        )
        return [dict(r) for r in rows]

    # ──────────── Global Analytics ────────────

    def global_stats(self) -> dict:
        """System-wide statistics across all sessions."""
        return {
            "total_sessions": self._db.fetch_value(
                "SELECT COUNT(*) FROM sessions") or 0,
            "total_score_records": self._db.fetch_value(
                "SELECT COUNT(*) FROM attention_scores") or 0,
            "total_alerts": self._db.fetch_value(
                "SELECT COUNT(*) FROM alerts") or 0,
            "total_students": self._db.fetch_value(
                "SELECT COUNT(DISTINCT student_id) FROM attention_scores") or 0,
            "overall_avg_score": self._db.fetch_value(
                "SELECT ROUND(AVG(score), 3) FROM attention_scores") or 0.0,
            "db_size_mb": self._db.get_db_size_mb(),
        }

