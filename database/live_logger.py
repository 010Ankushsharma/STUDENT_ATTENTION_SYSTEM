
"""
database/live_logger.py
High-performance batched database writer for real-time monitoring.

Buffers score records in memory and flushes to SQLite in batches
to minimise I/O overhead during live processing.

    Frame Loop
        │
        ▼
    logger.log_score(...)     ← Appends to in-memory buffer
        │
        ├─ buffer full?       → flush to DB (batch INSERT)
        ├─ timer expired?     → flush to DB
        └─ continue           → next frame

    End of session:
        logger.flush()        → Final flush
        logger.save_summary() → Aggregate & store session_summaries
"""
import time
from datetime import datetime
from typing import Optional, Dict, List
from .config import db_cfg
from .manager import DatabaseManager
from .session_repo import SessionRepository
from .student_repo import StudentRepository
from .score_repo import ScoreRepository
from .alert_repo import AlertRepository


class LiveLogger:
    """
    High-performance batched logger for live monitoring.

    Usage:
        db = DatabaseManager()
        logger = LiveLogger(db)

        session_id = logger.start_session("Math Class Period 3")

        # In your frame loop:
        logger.log_score(
            student_id=0, frame_num=42,
            score=0.85, state="attentive",
            ear=0.28, blink_rate=16.0, perclos=0.04,
            gaze_direction="center", drowsiness="alert",
            yaw=5.0, pitch=3.0, roll=1.0,
            head_direction="forward",
            hp_score=0.98, gaze_score=1.0,
            ear_score=1.0, blink_score=1.0, perclos_score=0.95,
        )

        logger.log_alert(
            student_id=2, alert_type="sleepy",
            severity="critical", message="Student 2 sleeping!",
            score=0.15, state="sleepy",
        )

        # End of session:
        logger.end_session(total_frames=5400, avg_score=0.72,
                           total_students=15)
    """

    def __init__(self, db: DatabaseManager):
        self._db = db
        self._session_repo = SessionRepository(db)
        self._student_repo = StudentRepository(db)
        self._score_repo = ScoreRepository(db)
        self._alert_repo = AlertRepository(db)

        # Buffer
        self._buffer: List[tuple] = []
        self._last_flush = time.time()
        self._session_id: Optional[str] = None

        # Frame skip tracking per student
        self._student_frame_counter: Dict[int, int] = {}

    # ──────────── Session Lifecycle ────────────

    def start_session(self, name: str = "Live Session",
                      camera_index: int = 0,
                      config: Optional[Dict] = None,
                      notes: str = "") -> str:
        """Start a new monitoring session. Returns session_id."""
        self._session_id = self._session_repo.create(
            name=name, camera_index=camera_index,
            config=config, notes=notes,
        )
        self._buffer.clear()
        self._student_frame_counter.clear()
        self._last_flush = time.time()
        return self._session_id

    def end_session(self, total_frames: int = 0,
                    avg_score: float = 0.0,
                    total_students: int = 0):
        """End the current session — flushes buffer and saves summary."""
        self.flush()
        if self._session_id:
            self._session_repo.end_session(
                self._session_id, total_frames, avg_score, total_students
            )
            self.save_summaries()
        self._session_id = None

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    # ──────────── Score Logging ────────────

    def log_score(self,
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
        """
        Log a single attention score record (buffered).

        Records are batched and flushed periodically for performance.
        Only logs every N frames per student (configurable).
        """
        if not self._session_id:
            return

        # Frame skip — only log every Nth frame per student
        counter = self._student_frame_counter.get(student_id, 0) + 1
        self._student_frame_counter[student_id] = counter
        if counter % db_cfg.log_every_n_frames != 0:
            return

        # Ensure student exists in DB
        self._student_repo.upsert(student_id)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        self._buffer.append((
            self._session_id, student_id, frame_num, now,
            round(score, 4), state,
            round(ear, 4), round(blink_rate, 1), round(perclos, 4),
            gaze_direction, drowsiness,
            round(yaw, 1), round(pitch, 1), round(roll, 1),
            head_direction,
            round(hp_score, 4), round(gaze_score, 4),
            round(ear_score, 4), round(blink_score, 4),
            round(perclos_score, 4),
        ))

        # Auto-flush if buffer full or time elapsed
        if (len(self._buffer) >= db_cfg.batch_size
                or (time.time() - self._last_flush) >= db_cfg.flush_interval_sec):
            self.flush()

    # ──────────── Alert Logging ────────────

    def log_alert(self,
                  student_id: int,
                  alert_type: str,
                  severity: str,
                  message: str,
                  score: float = 0.0,
                  state: str = "",
                  sustained_frames: int = 0):
        """Log an alert event (immediate write, not buffered)."""
        if not self._session_id:
            return
        self._alert_repo.insert(
            session_id=self._session_id,
            student_id=student_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            score=score,
            state=state,
            sustained_frames=sustained_frames,
        )

    # ──────────── Buffer Management ────────────

    def flush(self):
        """Flush buffered records to database."""
        if self._buffer:
            self._score_repo.insert_batch(self._buffer)
            self._buffer.clear()
        self._last_flush = time.time()

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    # ──────────── Session Summary ────────────

    def save_summaries(self):
        """
        Compute and store per-student session summaries.
        Called automatically at end_session().
        """
        if not self._session_id:
            return

        students = self._db.fetch_all(
            """SELECT DISTINCT student_id
               FROM attention_scores
               WHERE session_id = ?""",
            (self._session_id,)
        )

        for row in students:
            sid = row["student_id"]
            stats = self._db.fetch_one(
                """SELECT
                     ROUND(AVG(score), 4) AS avg_score,
                     ROUND(MIN(score), 4) AS min_score,
                     ROUND(MAX(score), 4) AS max_score,
                     COUNT(*) AS total_frames,
                     ROUND(AVG(ear), 4) AS avg_ear,
                     ROUND(AVG(blink_rate), 1) AS avg_blink_rate,
                     ROUND(AVG(ABS(yaw)), 1) AS avg_yaw,
                     ROUND(AVG(ABS(pitch)), 1) AS avg_pitch,
                     ROUND(
                       SUM(CASE WHEN state='attentive' THEN 1.0 ELSE 0 END)
                       / COUNT(*) * 100, 1
                     ) AS attentive_pct,
                     ROUND(
                       SUM(CASE WHEN state='distracted' THEN 1.0 ELSE 0 END)
                       / COUNT(*) * 100, 1
                     ) AS distracted_pct,
                     ROUND(
                       SUM(CASE WHEN state='sleepy' THEN 1.0 ELSE 0 END)
                       / COUNT(*) * 100, 1
                     ) AS sleepy_pct,
                     ROUND(
                       SUM(CASE WHEN state='looking_away' THEN 1.0 ELSE 0 END)
                       / COUNT(*) * 100, 1
                     ) AS away_pct
                   FROM attention_scores
                   WHERE session_id = ? AND student_id = ?""",
                (self._session_id, sid)
            )

            alert_count = self._db.fetch_value(
                "SELECT COUNT(*) FROM alerts WHERE session_id = ? AND student_id = ?",
                (self._session_id, sid)
            ) or 0

            if stats:
                self._db.execute(
                    """INSERT OR REPLACE INTO session_summaries
                       (session_id, student_id, avg_score, min_score, max_score,
                        attentive_pct, distracted_pct, sleepy_pct, away_pct,
                        total_frames, total_alerts, avg_ear, avg_blink_rate,
                        avg_yaw, avg_pitch)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (self._session_id, sid,
                     stats["avg_score"], stats["min_score"], stats["max_score"],
                     stats["attentive_pct"], stats["distracted_pct"],
                     stats["sleepy_pct"], stats["away_pct"],
                     stats["total_frames"], alert_count,
                     stats["avg_ear"], stats["avg_blink_rate"],
                     stats["avg_yaw"], stats["avg_pitch"])
                )

    def reset(self):
        """Clear buffer without flushing."""
        self._buffer.clear()
        self._student_frame_counter.clear()

