# 🏗️ Architecture — Student Attention Detection System

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                                     │
│                                                                              │
│   main_app.py ─── StudentAttentionSystem                                    │
│       │                                                                      │
│       ├── run()           Main event loop + keyboard handling                │
│       ├── process_frame() Single-frame pipeline orchestration                │
│       └── _shutdown()     Graceful cleanup                                   │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                         PROCESSING LAYER                                      │
│                                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│   │   face_      │  │   eye_       │  │   head_      │  │  attention_  │  │
│   │   detection  │  │   tracking   │  │   pose       │  │  scoring     │  │
│   │              │  │              │  │              │  │              │  │
│   │  Preprocessor│  │  EAR Calc    │  │  Landmark    │  │  Signal      │  │
│   │  Detector    │  │  Blink Det   │  │  Mapper      │  │  Fusion      │  │
│   │  Tracker     │  │  Drowsiness  │  │  Pose Calc   │  │  Scorer      │  │
│   │  Annotator   │  │  Gaze Est    │  │  Direction   │  │  Classifier  │  │
│   │              │  │  Visualizer  │  │  Attention   │  │  Alert       │  │
│   │              │  │              │  │  Visualizer  │  │  Engine      │  │
│   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                         INFRASTRUCTURE LAYER                                  │
│                                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────────┐ │
│   │   database   │  │   dashboard  │  │            core                   │ │
│   │              │  │              │  │                                   │ │
│   │  Manager     │  │  API (FastAPI)│ │  Logger     (rotating files)     │ │
│   │  Repos       │  │  Server      │  │  Errors     (exception tree)     │ │
│   │  Analytics   │  │  WebSocket   │  │  Performance(timer, FPS, perf)   │ │
│   │  LiveLogger  │  │  Templates   │  │  Config     (env loader)         │ │
│   └──────────────┘  └──────────────┘  └──────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Single Responsibility
Each module handles exactly one concern:
- `face_detection/` — Only detects and tracks faces
- `eye_tracking/` — Only analyzes eye metrics
- `head_pose/` — Only estimates head orientation
- `attention_scoring/` — Only computes attention from signals
- `database/` — Only persists and queries data
- `dashboard/` — Only serves the web interface

### 2. Dependency Inversion
Modules don't import each other. The main app orchestrates:
```python
# main_app.py composes everything
face_result = face_detector.detect(frame)
eye_result = eye_tracker.analyze(landmarks)
pose_result = pose_calc.compute(landmarks)
score_result = scoring.update(eye_result, pose_result)
db_logger.log(score_result)
```

### 3. Error Isolation
Each module has try/except boundaries. A failure in eye tracking
doesn't crash head pose or scoring — the system degrades gracefully.

### 4. Performance by Design
- Batched DB writes (50 records at once)
- Frame skipping (configurable)
- Per-student lazy initialization
- Rolling windows instead of growing arrays
- JPEG compression for dashboard streaming

---

## Data Flow

```
Frame (BGR numpy array)
    │
    ▼ preprocess (~2ms)
Resized + CLAHE frame
    │
    ├──▶ face_detector.detect() (~8ms)
    │       └── List[BBox]
    │           └── tracker.update() → Dict[id, Student]
    │
    └──▶ face_mesh.process() (~12ms)
            └── List[FaceLandmarks]
                    │
                    │ (matched to tracked students by centroid proximity)
                    │
                    ├──▶ eye_tracking per student (~3ms)
                    │       EAR → blink → drowsiness → gaze
                    │       Output: ear, blink_rate, perclos, gaze_dir, drowsy_level
                    │
                    └──▶ head_pose per student (~2ms)
                            landmarks → solvePnP → Euler → direction
                            Output: yaw, pitch, roll, head_direction

                    Combined signals
                         │
                         ▼ scoring.update() (~1ms)
                    ┌─────────────────────────┐
                    │  Signal Fusion           │
                    │  HP×0.35 + Gaze×0.20    │
                    │  + EAR×0.20 + Blink×0.10│
                    │  + PERCLOS×0.15          │
                    │         │                │
                    │         ▼                │
                    │  EMA Smoothing (α=0.3)   │
                    │         │                │
                    │         ▼                │
                    │  State Classification    │
                    │  + Hysteresis            │
                    │         │                │
                    │         ▼                │
                    │  Alert Check             │
                    └─────────┬───────────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
               Visualize   Database   Dashboard
               (OpenCV)    (SQLite)   (WebSocket)
```

---

## Threading Model

```
┌─────────────────────────────────────┐
│         MAIN THREAD                  │
│   Camera capture + CV processing     │
│   + OpenCV window                    │
└───────────────────┬─────────────────┘
                    │ frame bytes
                    ▼
┌─────────────────────────────────────┐
│      DASHBOARD THREAD (uvicorn)      │
│   FastAPI server                     │
│   - REST endpoints                   │
│   - WebSocket push (every 500ms)     │
│   - MJPEG stream (15fps)             │
└─────────────────────────────────────┘
```

---

## Database Design

### Write Path (hot path — optimized for speed):
1. `LiveLogger.log_score()` → append to memory buffer
2. Buffer reaches 50 records OR 3 seconds elapsed → batch INSERT
3. Alerts are written immediately (rare events)

### Read Path (analytics — optimized for insight):
- Indexed queries on session_id + student_id
- Pre-aggregated session_summaries table
- WAL mode for concurrent read/write

---

## Configuration Hierarchy

```
Priority (highest to lowest):
  1. CLI arguments         (--camera 1 --port 8080)
  2. Environment variables (CAMERA_INDEX=1)
  3. .env file             (CAMERA_INDEX=1)
  4. Code defaults         (camera_index=0)
```

---

## Error Handling Strategy

```
Level 1: Per-frame recovery
    If eye tracking fails for one student → use defaults → continue

Level 2: Per-student isolation
    One student's error doesn't affect others

Level 3: Module degradation
    If DB fails → run without persistence
    If dashboard fails → run without web UI

Level 4: Fatal errors
    Camera disconnection → graceful shutdown with session save
```