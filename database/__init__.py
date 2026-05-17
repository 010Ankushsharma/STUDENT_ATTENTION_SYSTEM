
"""
Database Integration Module
============================
SQLite-backed persistence for the Student Attention Detection System.

Core Components:
    - DatabaseManager   : Connection pool, schema bootstrap, migrations
    - SessionRepository : CRUD for monitoring sessions
    - StudentRepository : CRUD for student profiles & enrolment
    - ScoreRepository   : Insert / query attention score snapshots
    - AlertRepository   : Store and retrieve alert events
    - AnalyticsEngine   : Pre-built analytical queries & reports
    - LiveLogger        : High-perf batched writer for real-time data
    - DBConfig          : All tunable database parameters

Usage:
    from database import DatabaseManager, LiveLogger, AnalyticsEngine

    db = DatabaseManager()          # auto-creates tables
    logger = LiveLogger(db)         # batch-insert every N frames
    analytics = AnalyticsEngine(db) # run reports

    # In your frame loop:
    logger.log_score(session_id, student_id, score, state, ...)
    logger.flush()                  # at end of session

    # Analytics:
    report = analytics.student_summary(student_id, session_id)
    timeline = analytics.attention_timeline(session_id)
"""
from .config import DBConfig, db_cfg
from .manager import DatabaseManager
from .session_repo import SessionRepository
from .student_repo import StudentRepository
from .score_repo import ScoreRepository
from .alert_repo import AlertRepository
from .analytics import AnalyticsEngine
from .live_logger import LiveLogger

__all__ = [
    "DBConfig",
    "db_cfg",
    "DatabaseManager",
    "SessionRepository",
    "StudentRepository",
    "ScoreRepository",
    "AlertRepository",
    "AnalyticsEngine",
    "LiveLogger",
]
__version__ = "1.0.0"