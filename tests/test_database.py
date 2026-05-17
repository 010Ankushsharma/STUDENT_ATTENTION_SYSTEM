
"""
tests/test_database.py
Comprehensive tests for the database integration module.
"""
import os
import pytest
import time
from database.config import db_cfg
from database.manager import DatabaseManager
from database.session_repo import SessionRepository
from database.student_repo import StudentRepository
from database.score_repo import ScoreRepository
from database.alert_repo import AlertRepository
from database.analytics import AnalyticsEngine
from database.live_logger import LiveLogger


TEST_DB = "test_attention.db"


@pytest.fixture(autouse=True)
def setup_teardown():
    """Create fresh test DB before each test, remove after."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@pytest.fixture
def db():
    return DatabaseManager(TEST_DB)


@pytest.fixture
def session_repo(db):
    return SessionRepository(db)


@pytest.fixture
def student_repo(db):
    return StudentRepository(db)


@pytest.fixture
def score_repo(db):
    return ScoreRepository(db)


@pytest.fixture
def alert_repo(db):
    return AlertRepository(db)


@pytest.fixture
def analytics(db):
    return AnalyticsEngine(db)


@pytest.fixture
def logger(db):
    return LiveLogger(db)


# ────────── DatabaseManager Tests ──────────

class TestDatabaseManager:
    def test_tables_created(self, db):
        tables = db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        names = {r["name"] for r in tables}
        assert "sessions" in names
        assert "students" in names
        assert "attention_scores" in names
        assert "alerts" in names
        assert "session_summaries" in names
        assert "schema_version" in names

    def test_indexes_created(self, db):
        indexes = db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        names = {r["name"] for r in indexes}
        assert "idx_scores_session" in names
        assert "idx_scores_student" in names
        assert "idx_alerts_session" in names

    def test_schema_version(self, db):
        v = db.get_schema_version()
        assert v == db_cfg.schema_version

    def test_table_counts_empty(self, db):
        counts = db.get_table_counts()
        assert counts["sessions"] == 0
        assert counts["attention_scores"] == 0

    def test_execute_and_fetch(self, db):
        db.execute(
            "INSERT INTO students (student_track_id, label) VALUES (?, ?)",
            (1, "Test")
        )
        row = db.fetch_one(
            "SELECT * FROM students WHERE student_track_id = ?", (1,)
        )
        assert row["label"] == "Test"

    def test_fetch_value(self, db):
        db.execute(
            "INSERT INTO students (student_track_id, label) VALUES (?, ?)",
            (1, "Test")
        )
        count = db.fetch_value("SELECT COUNT(*) FROM students")
        assert count == 1


# ────────── SessionRepository Tests ──────────

class TestSessionRepository:
    def test_create_session(self, session_repo):
        sid = session_repo.create("Test Session", camera_index=0)
        assert sid is not None
        assert len(sid) > 0

    def test_get_session(self, session_repo):
        sid = session_repo.create("My Session")
        s = session_repo.get(sid)
        assert s["name"] == "My Session"
        assert s["status"] == "active"

    def test_end_session(self, session_repo):
        sid = session_repo.create("Session")
        session_repo.end_session(sid, total_frames=100, avg_score=0.75)
        s = session_repo.get(sid)
        assert s["status"] == "completed"
        assert s["total_frames"] == 100

    def test_list_recent(self, session_repo):
        session_repo.create("S1")
        session_repo.create("S2")
        sessions = session_repo.list_recent()
        assert len(sessions) == 2

    def test_delete(self, session_repo):
        sid = session_repo.create("To Delete")
        session_repo.delete(sid)
        assert session_repo.get(sid) is None

    def test_count(self, session_repo):
        session_repo.create("S1")
        session_repo.create("S2")
        assert session_repo.count() == 2


# ────────── StudentRepository Tests ──────────

class TestStudentRepository:
    def test_upsert_new(self, student_repo):
        db_id = student_repo.upsert(track_id=5, label="Row 1 Seat 5")
        assert db_id > 0

    def test_upsert_existing(self, student_repo):
        id1 = student_repo.upsert(track_id=5, label="First")
        id2 = student_repo.upsert(track_id=5, label="Updated")
        assert id1 == id2

    def test_get_by_track_id(self, student_repo):
        student_repo.upsert(track_id=3, label="Student 3")
        s = student_repo.get_by_track_id(3)
        assert s["label"] == "Student 3"

    def test_list_all(self, student_repo):
        student_repo.upsert(1)
        student_repo.upsert(2)
        assert len(student_repo.list_all()) == 2

    def test_set_label(self, student_repo):
        student_repo.upsert(1, label="Old")
        student_repo.set_label(1, "New Label")
        s = student_repo.get_by_track_id(1)
        assert s["label"] == "New Label"

    def test_count(self, student_repo):
        student_repo.upsert(1)
        student_repo.upsert(2)
        assert student_repo.count() == 2


# ────────── ScoreRepository Tests ──────────

class TestScoreRepository:
    def test_insert_single(self, score_repo, session_repo):
        sid = session_repo.create("Test")
        score_repo.insert(sid, student_id=0, frame_num=1,
                         score=0.85, state="attentive")
        assert score_repo.count(sid) == 1

    def test_insert_batch(self, score_repo, session_repo):
        sid = session_repo.create("Test")
        records = []
        for i in range(50):
            from datetime import datetime
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            records.append((
                sid, 0, i, now, 0.8, "attentive",
                0.28, 16.0, 0.04, "center", "alert",
                5.0, 3.0, 1.0, "forward",
                0.98, 1.0, 1.0, 1.0, 0.95,
            ))
        score_repo.insert_batch(records)
        assert score_repo.count(sid) == 50

    def test_get_student_history(self, score_repo, session_repo):
        sid = session_repo.create("Test")
        for i in range(5):
            score_repo.insert(sid, 0, i, 0.8 + i*0.02, "attentive")
        history = score_repo.get_student_history(sid, 0)
        assert len(history) == 5

    def test_count_by_state(self, score_repo, session_repo):
        sid = session_repo.create("Test")
        score_repo.insert(sid, 0, 1, 0.9, "attentive")
        score_repo.insert(sid, 0, 2, 0.5, "distracted")
        score_repo.insert(sid, 0, 3, 0.2, "sleepy")
        counts = score_repo.count_by_state(sid)
        assert counts["attentive"] == 1
        assert counts["distracted"] == 1


# ────────── AlertRepository Tests ──────────

class TestAlertRepository:
    def test_insert_alert(self, alert_repo, session_repo):
        sid = session_repo.create("Test")
        alert_repo.insert(sid, 0, "sleepy", "critical", "Student 0 sleeping!")
        assert alert_repo.count(sid) == 1

    def test_get_session_alerts(self, alert_repo, session_repo):
        sid = session_repo.create("Test")
        alert_repo.insert(sid, 0, "sleepy", "critical", "Alert 1")
        alert_repo.insert(sid, 1, "looking_away", "warning", "Alert 2")
        alerts = alert_repo.get_session_alerts(sid)
        assert len(alerts) == 2

    def test_acknowledge(self, alert_repo, session_repo):
        sid = session_repo.create("Test")
        alert_repo.insert(sid, 0, "sleepy", "critical", "Alert")
        alerts = alert_repo.get_session_alerts(sid)
        alert_repo.acknowledge(alerts[0]["id"])
        updated = alert_repo.get_session_alerts(sid)
        assert updated[0]["acknowledged"] == 1

    def test_filter_by_severity(self, alert_repo, session_repo):
        sid = session_repo.create("Test")
        alert_repo.insert(sid, 0, "sleepy", "critical", "C1")
        alert_repo.insert(sid, 1, "low_attention", "warning", "W1")
        crits = alert_repo.get_by_severity(sid, "critical")
        assert len(crits) == 1


# ────────── AnalyticsEngine Tests ──────────

class TestAnalyticsEngine:
    def _populate(self, session_repo, score_repo, alert_repo):
        sid = session_repo.create("Analytics Test")
        for i in range(100):
            state = "attentive" if i % 3 != 0 else "distracted"
            score_repo.insert(sid, student_id=0, frame_num=i,
                             score=0.8 if state == "attentive" else 0.5,
                             state=state, ear=0.28, blink_rate=16.0,
                             perclos=0.04, yaw=5.0, pitch=3.0)
        for i in range(50):
            score_repo.insert(sid, student_id=1, frame_num=i,
                             score=0.3, state="sleepy",
                             ear=0.15, blink_rate=28.0, perclos=0.20,
                             yaw=-5.0, pitch=-10.0)
        alert_repo.insert(sid, 1, "sleepy", "critical", "Student 1 drowsy")
        return sid

    def test_session_overview(self, analytics, session_repo, score_repo, alert_repo):
        sid = self._populate(session_repo, score_repo, alert_repo)
        overview = analytics.session_overview(sid)
        assert overview["total_students"] == 2
        assert overview["total_score_records"] == 150
        assert overview["alert_count"] == 1

    def test_class_ranking(self, analytics, session_repo, score_repo, alert_repo):
        sid = self._populate(session_repo, score_repo, alert_repo)
        ranking = analytics.class_ranking(sid)
        assert ranking[0]["rank"] == 1
        assert ranking[0]["avg_score"] > ranking[1]["avg_score"]

    def test_student_summary(self, analytics, session_repo, score_repo, alert_repo):
        sid = self._populate(session_repo, score_repo, alert_repo)
        summary = analytics.student_summary(sid, 0)
        assert summary["total_frames"] == 100
        assert "attentive" in summary["state_distribution"]

    def test_state_distribution(self, analytics, session_repo, score_repo, alert_repo):
        sid = self._populate(session_repo, score_repo, alert_repo)
        dist = analytics.state_distribution(sid)
        assert "attentive" in dist or "distracted" in dist

    def test_students_needing_attention(self, analytics, session_repo, score_repo, alert_repo):
        sid = self._populate(session_repo, score_repo, alert_repo)
        needy = analytics.students_needing_attention(sid, threshold=0.5)
        assert len(needy) >= 1  # Student 1 (avg 0.3) should be here

    def test_global_stats(self, analytics, session_repo, score_repo, alert_repo):
        self._populate(session_repo, score_repo, alert_repo)
        stats = analytics.global_stats()
        assert stats["total_sessions"] >= 1
        assert stats["total_score_records"] >= 150


# ────────── LiveLogger Tests ──────────

class TestLiveLogger:
    def test_start_and_end_session(self, logger):
        sid = logger.start_session("Test Logger Session")
        assert sid is not None
        logger.end_session(total_frames=100, avg_score=0.8)

    def test_log_score_buffered(self, logger):
        db_cfg.log_every_n_frames = 1  # Log every frame for testing
        logger.start_session("Buffered Test")
        for i in range(10):
            logger.log_score(student_id=0, frame_num=i,
                           score=0.85, state="attentive")
        assert logger.buffer_size > 0  # Not yet flushed

    def test_flush(self, logger, db):
        db_cfg.log_every_n_frames = 1
        sid = logger.start_session("Flush Test")
        for i in range(10):
            logger.log_score(student_id=0, frame_num=i,
                           score=0.85, state="attentive")
        logger.flush()
        count = db.fetch_value(
            "SELECT COUNT(*) FROM attention_scores WHERE session_id = ?",
            (sid,)
        )
        assert count == 10

    def test_auto_flush_on_batch_size(self, logger, db):
        db_cfg.log_every_n_frames = 1
        db_cfg.batch_size = 5
        sid = logger.start_session("Auto Flush")
        for i in range(6):
            logger.log_score(student_id=0, frame_num=i,
                           score=0.85, state="attentive")
        # After 5 records, should have auto-flushed
        count = db.fetch_value(
            "SELECT COUNT(*) FROM attention_scores WHERE session_id = ?",
            (sid,)
        )
        assert count >= 5
        db_cfg.batch_size = 50  # Restore

    def test_log_alert(self, logger, db):
        sid = logger.start_session("Alert Test")
        logger.log_alert(student_id=2, alert_type="sleepy",
                        severity="critical", message="Sleeping!")
        count = db.fetch_value(
            "SELECT COUNT(*) FROM alerts WHERE session_id = ?", (sid,)
        )
        assert count == 1

    def test_save_summaries(self, logger, db):
        db_cfg.log_every_n_frames = 1
        sid = logger.start_session("Summary Test")
        for i in range(20):
            logger.log_score(student_id=0, frame_num=i,
                           score=0.85, state="attentive",
                           ear=0.28, blink_rate=16.0)
        logger.end_session(total_frames=20, avg_score=0.85)
        summaries = db.fetch_all(
            "SELECT * FROM session_summaries WHERE session_id = ?",
            (sid,)
        )
        assert len(summaries) == 1
        assert summaries[0]["avg_score"] > 0

    def test_frame_skip(self, logger, db):
        """Only logs every Nth frame per student."""
        db_cfg.log_every_n_frames = 5
        sid = logger.start_session("Skip Test")
        for i in range(20):
            logger.log_score(student_id=0, frame_num=i,
                           score=0.85, state="attentive")
        logger.flush()
        count = db.fetch_value(
            "SELECT COUNT(*) FROM attention_scores WHERE session_id = ?",
            (sid,)
        )
        assert count == 4  # frames 5, 10, 15, 20
        db_cfg.log_every_n_frames = 5  # Restore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
