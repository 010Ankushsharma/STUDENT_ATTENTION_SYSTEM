
"""
database/schema.py
Complete database schema for the Student Attention Detection System.

Schema Diagram:

    ┌──────────────────────┐       ┌──────────────────────────┐
    │      sessions         │       │       students            │
    ├──────────────────────┤       ├──────────────────────────┤
    │ PK session_id  TEXT   │       │ PK student_db_id INTEGER  │
    │    name         TEXT   │◄──┐  │    student_track_id INT   │
    │    start_time   TEXT   │   │  │    label          TEXT    │
    │    end_time     TEXT   │   │  │    first_seen     TEXT    │
    │    camera_index INT    │   │  │    created_at     TEXT    │
    │    status       TEXT   │   │  └──────────────────────────┘
    │    total_frames INT    │   │
    │    config_json  TEXT   │   │
    │    created_at   TEXT   │   │
    └──────────┬───────────┘   │
               │               │
    ┌──────────▼───────────┐   │  ┌──────────────────────────────┐
    │  attention_scores     │   │  │        alerts                 │
    ├──────────────────────┤   │  ├──────────────────────────────┤
    │ PK id      INTEGER    │   │  │ PK id          INTEGER       │
    │ FK session_id TEXT  ──┼───┤  │ FK session_id   TEXT  ───────┤
    │    student_id  INT    │   │  │    student_id    INT         │
    │    frame_num   INT    │   │  │    alert_type    TEXT        │
    │    timestamp    TEXT   │   │  │    severity      TEXT        │
    │    score       REAL    │   │  │    message       TEXT        │
    │    state       TEXT    │   │  │    score         REAL        │
    │    ear         REAL    │   │  │    state         TEXT        │
    │    blink_rate  REAL    │   │  │    timestamp     TEXT        │
    │    perclos     REAL    │   └──│    created_at    TEXT        │
    │    gaze_dir    TEXT    │      └──────────────────────────────┘
    │    yaw         REAL    │
    │    pitch       REAL    │      ┌──────────────────────────────┐
    │    drowsiness  TEXT    │      │    session_summaries          │
    │    head_dir    TEXT    │      ├──────────────────────────────┤
    │    hp_score    REAL    │      │ PK id          INTEGER       │
    │    gaze_score  REAL    │      │ FK session_id   TEXT         │
    │    ear_score   REAL    │      │    student_id    INT         │
    │    blink_score REAL    │      │    avg_score     REAL        │
    │    perc_score  REAL    │      │    attentive_pct REAL        │
    │    created_at  TEXT    │      │    distracted_pct REAL       │
    └──────────────────────┘      │    sleepy_pct    REAL        │
                                   │    away_pct      REAL        │
                                   │    total_frames  INT         │
                                   │    total_blinks  INT         │
                                   │    total_alerts  INT         │
                                   │    created_at    TEXT        │
                                   └──────────────────────────────┘

Indexes:
    - idx_scores_session        ON attention_scores(session_id)
    - idx_scores_student        ON attention_scores(student_id)
    - idx_scores_session_student ON attention_scores(session_id, student_id)
    - idx_scores_timestamp      ON attention_scores(timestamp)
    - idx_alerts_session        ON alerts(session_id)
    - idx_alerts_student        ON alerts(student_id)
"""

SCHEMA_SQL = """
-- ═══════════════════════════════════════════════
-- TABLE: sessions
-- Stores metadata for each monitoring session
-- ═══════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    name            TEXT NOT NULL DEFAULT 'Unnamed Session',
    start_time      TEXT NOT NULL,
    end_time        TEXT,
    camera_index    INTEGER DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'paused', 'completed', 'aborted')),
    total_frames    INTEGER DEFAULT 0,
    avg_class_score REAL DEFAULT 0.0,
    total_students  INTEGER DEFAULT 0,
    config_json     TEXT,
    notes           TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);


-- ═══════════════════════════════════════════════
-- TABLE: students
-- Persistent student profiles across sessions
-- ═══════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS students (
    student_db_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    student_track_id    INTEGER NOT NULL,
    label               TEXT DEFAULT '',
    first_seen          TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    total_sessions      INTEGER DEFAULT 0,
    avg_attention_pct   REAL DEFAULT 0.0,
    created_at          TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);


-- ═══════════════════════════════════════════════
-- TABLE: attention_scores
-- Per-frame attention data (the main data table)
-- ═══════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS attention_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    student_id      INTEGER NOT NULL,
    frame_num       INTEGER NOT NULL,
    timestamp       TEXT NOT NULL,

    -- Core attention
    score           REAL NOT NULL,
    state           TEXT NOT NULL
                    CHECK (state IN ('attentive','distracted','sleepy','looking_away')),

    -- Eye tracking signals
    ear             REAL,
    blink_rate      REAL,
    perclos         REAL,
    gaze_direction  TEXT,
    drowsiness      TEXT,

    -- Head pose signals
    yaw             REAL,
    pitch           REAL,
    roll            REAL,
    head_direction  TEXT,

    -- Component scores (from signal fusion)
    hp_score        REAL,
    gaze_score      REAL,
    ear_score       REAL,
    blink_score     REAL,
    perclos_score   REAL,

    created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),

    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        ON DELETE CASCADE
);


-- ═══════════════════════════════════════════════
-- TABLE: alerts
-- Alert events triggered during monitoring
-- ═══════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    student_id      INTEGER NOT NULL,
    alert_type      TEXT NOT NULL
                    CHECK (alert_type IN ('low_attention','sleepy',
                           'looking_away','rapid_decline')),
    severity        TEXT NOT NULL
                    CHECK (severity IN ('warning', 'critical')),
    message         TEXT NOT NULL,
    score           REAL,
    state           TEXT,
    sustained_frames INTEGER DEFAULT 0,
    acknowledged    INTEGER DEFAULT 0,
    timestamp       TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),

    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        ON DELETE CASCADE
);


-- ═══════════════════════════════════════════════
-- TABLE: session_summaries
-- Aggregated end-of-session stats per student
-- ═══════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS session_summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    student_id      INTEGER NOT NULL,
    avg_score       REAL DEFAULT 0.0,
    min_score       REAL DEFAULT 0.0,
    max_score       REAL DEFAULT 1.0,
    attentive_pct   REAL DEFAULT 0.0,
    distracted_pct  REAL DEFAULT 0.0,
    sleepy_pct      REAL DEFAULT 0.0,
    away_pct        REAL DEFAULT 0.0,
    total_frames    INTEGER DEFAULT 0,
    total_blinks    INTEGER DEFAULT 0,
    total_alerts    INTEGER DEFAULT 0,
    avg_ear         REAL DEFAULT 0.0,
    avg_blink_rate  REAL DEFAULT 0.0,
    avg_yaw         REAL DEFAULT 0.0,
    avg_pitch       REAL DEFAULT 0.0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),

    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        ON DELETE CASCADE,
    UNIQUE(session_id, student_id)
);


-- ═══════════════════════════════════════════════
-- TABLE: schema_version
-- Track database schema version for migrations
-- ═══════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);


-- ═══════════════════════════════════════════════
-- INDEXES for query performance
-- ═══════════════════════════════════════════════
CREATE INDEX IF NOT EXISTS idx_scores_session
    ON attention_scores(session_id);

CREATE INDEX IF NOT EXISTS idx_scores_student
    ON attention_scores(student_id);

CREATE INDEX IF NOT EXISTS idx_scores_session_student
    ON attention_scores(session_id, student_id);

CREATE INDEX IF NOT EXISTS idx_scores_timestamp
    ON attention_scores(timestamp);

CREATE INDEX IF NOT EXISTS idx_scores_state
    ON attention_scores(state);

CREATE INDEX IF NOT EXISTS idx_alerts_session
    ON alerts(session_id);

CREATE INDEX IF NOT EXISTS idx_alerts_student
    ON alerts(student_id);

CREATE INDEX IF NOT EXISTS idx_alerts_type
    ON alerts(alert_type);

CREATE INDEX IF NOT EXISTS idx_summaries_session
    ON session_summaries(session_id);

CREATE INDEX IF NOT EXISTS idx_summaries_student
    ON session_summaries(student_id);
"""
