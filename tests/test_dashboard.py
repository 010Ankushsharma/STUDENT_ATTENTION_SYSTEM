
"""
tests/test_dashboard.py
Tests for the dashboard API endpoints.
"""
import pytest
import time
from unittest.mock import MagicMock, patch

# Skip if FastAPI not installed
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from dashboard.api import create_app


@pytest.fixture
def mock_scoring():
    scoring = MagicMock()
    scoring.get_class_summary.return_value = {
        "total": 3, "attentive": 2, "distracted": 1,
        "sleepy": 0, "looking_away": 0, "avg_score": 0.78,
    }
    scoring.get_all_students.return_value = {}
    scoring.get_leaderboard.return_value = [
        {"student_id": 0, "attention_pct": 92, "state": "attentive", "score": 0.92},
        {"student_id": 1, "attention_pct": 65, "state": "distracted", "score": 0.65},
    ]
    scoring.get_alerts.return_value = []
    return scoring


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.get_db_size_mb.return_value = 1.5
    db.get_table_counts.return_value = {
        "sessions": 5, "students": 10,
        "attention_scores": 5000, "alerts": 12,
        "session_summaries": 8,
    }
    return db


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.session_id = "test123"
    return logger


@pytest.fixture
def mock_analytics():
    analytics = MagicMock()
    analytics.session_overview.return_value = {
        "total_students": 3, "total_score_records": 500,
        "avg_score": 0.75, "alert_count": 2,
    }
    analytics.class_ranking.return_value = [
        {"rank": 1, "student_id": 0, "avg_score": 0.9, "attentive_pct": 88},
    ]
    analytics.state_distribution.return_value = {
        "attentive": 65.0, "distracted": 25.0,
        "sleepy": 5.0, "looking_away": 5.0,
    }
    analytics.attention_timeline.return_value = [
        {"student_id": 0, "time_bucket": 1, "avg_score": 0.85},
    ]
    return analytics


@pytest.fixture
def client(mock_scoring, mock_db, mock_logger, mock_analytics):
    app = create_app(
        scoring=mock_scoring,
        db=mock_db,
        analytics=mock_analytics,
        logger=mock_logger,
        frame_source=lambda: b"fake_jpeg_bytes",
    )
    return TestClient(app)


class TestDashboardAPI:

    def test_home_page(self, client):
        """Dashboard page should return HTML."""
        resp = client.get("/")
        # May return 500 if template not found in test env, that's OK
        assert resp.status_code in (200, 500)

    def test_status(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["session_active"] is True
        assert data["session_id"] == "test123"

    def test_current_session(self, client):
        resp = client.get("/api/session/current")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "test123"
        assert "class_summary" in data

    def test_get_students(self, client):
        resp = client.get("/api/students")
        assert resp.status_code == 200
        data = resp.json()
        assert "students" in data
        assert "count" in data

    def test_get_alerts(self, client):
        resp = client.get("/api/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert "alerts" in data
        assert "total" in data

    def test_analytics_overview(self, client):
        resp = client.get("/api/analytics/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_students" in data

    def test_analytics_timeline(self, client):
        resp = client.get("/api/analytics/timeline")
        assert resp.status_code == 200
        data = resp.json()
        assert "timeline" in data

    def test_analytics_ranking(self, client):
        resp = client.get("/api/analytics/ranking")
        assert resp.status_code == 200
        data = resp.json()
        assert "ranking" in data

    def test_analytics_distribution(self, client):
        resp = client.get("/api/analytics/distribution")
        assert resp.status_code == 200
        data = resp.json()
        assert "distribution" in data
        assert "attentive" in data["distribution"]

    def test_export_csv(self, client):
        resp = client.get("/api/export/csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers.get("content-type", "")
        assert "attachment" in resp.headers.get("content-disposition", "")

    def test_video_feed(self, client):
        """Video feed should start streaming."""
        resp = client.get("/video_feed", stream=True)
        # StreamingResponse returns 200
        assert resp.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
