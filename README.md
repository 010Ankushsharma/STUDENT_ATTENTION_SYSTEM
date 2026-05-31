
# 🎓 Student Attention Detection System v1.0.0

**Complete AI-powered real-time classroom attention monitoring system.**
   
7 integrated modules • SQLite database • Web dashboard • Docker-ready    

---
  
## ⚡ One-Command Start

```bash
pip install -r requirements.txt && python run_system.py
```

Then open **http://localhost:8000** for the live dashboard.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STUDENT ATTENTION SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   WEBCAM → Preprocess → FACE DETECTION → Assign Student IDs     │
│                                │                                 │
│                    ┌───────────┼───────────┐                    │
│                    ▼           ▼           ▼                    │
│              EYE TRACKING  BLINK DET.  HEAD POSE                │
│              • EAR         • Frequency • solvePnP               │
│              • Gaze        • PERCLOS   • Direction               │
│              • Drowsiness  • Rate      • Attention              │
│                    │           │           │                    │
│                    └───────────┼───────────┘                    │
│                                ▼                                 │
│                    ATTENTION SCORING                              │
│                    • 5-signal weighted fusion                     │
│                    • EMA temporal smoothing                       │
│                    • 4-state classification                       │
│                    • Alert engine                                 │
│                                │                                 │
│                    ┌───────────┼───────────┐                    │
│                    ▼           ▼           ▼                    │
│               DATABASE    DASHBOARD    VISUALIZER               │
│               (SQLite)    (FastAPI)    (OpenCV)                  │
│               • Sessions  • WebSocket  • Overlays               │
│               • Scores    • Charts     • Score bars             │
│               • Analytics • Export     • Alert flash            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure (88 files)

```
student_attention_system/
│
├── 🚀 main_app.py              ← Unified application entry point
├── ⚙️  main_config.py           ← Global configuration
├── ▶️  run_system.py             ← One-click launcher
├── 📦 requirements.txt          ← All dependencies
├── 🐳 Dockerfile               ← Docker deployment
├── 🐳 docker-compose.yml       ← Docker Compose
├── 📋 setup.py                 ← pip install -e .
├── 🔧 Makefile                 ← Common commands
├── 🔒 .env.example             ← Environment template
├── 🚫 .gitignore
│
├── core/                        # 🔧 Shared Utilities
│   ├── logger.py                #   Coloured logs + file rotation
│   ├── errors.py                #   Exception hierarchy (6 types)
│   ├── performance.py           #   FPS counter + Timer + Monitor
│   └── config_loader.py         #   .env + env var loading
│
├── face_detection/              # 📸 Module 1 (9 files)
├── eye_tracking/                # 👁️ Module 2 (9 files) — includes blink detection
├── head_pose/                   # 🧭 Module 3 (9 files)
├── attention_scoring/           # 🧠 Module 4 (9 files)
├── database/                    # 🗄️ Module 5 (10 files)
├── dashboard/                   # 🌐 Module 6 (6 files)
│
├── tests/                       # ✅ 13 test files, 130+ cases
│   ├── test_face_detection.py
│   ├── test_ear_calculator.py
│   ├── test_blink_detector.py
│   ├── test_drowsiness.py
│   ├── test_pose_calculator.py
│   ├── test_direction_classifier.py
│   ├── test_signal_fusion.py
│   ├── test_state_classifier.py
│   ├── test_alert_engine.py
│   ├── test_pipeline.py
│   ├── test_database.py
│   ├── test_dashboard.py
│   └── test_core.py
│
└── docs/                        # 📚 Documentation
    ├── SETUP_GUIDE.md           #   Installation & configuration
    ├── ARCHITECTURE.md          #   System design & data flow
    ├── DEPLOYMENT.md            #   Docker, systemd, nginx
    ├── face_detection_README.md
    ├── eye_tracking_README.md
    ├── head_pose_README.md
    └── attention_scoring_README.md
```

---

## 🌐 Web Dashboard

| Feature | Description |
|---------|-------------|
| 📹 Live Video | MJPEG annotated webcam stream |
| 📊 Real-time Scores | Per-student attention % (WebSocket) |
| 📈 Timeline Chart | Class attention over time |
| 🍩 Distribution | State breakdown pie chart |
| 🏆 Leaderboard | Students ranked by attention |
| 🚨 Alerts | Live notification feed |
| 📥 Export CSV | One-click download |
| 📁 Export Excel | Multi-sheet workbook |

**14 REST endpoints + WebSocket + MJPEG stream**

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `d` | Toggle overlay dashboard |
| `s` | Save screenshot |
| `r` | Reset all trackers |
| `p` | Print class summary |
| `a` | Print DB analytics |
| `w` | Open web dashboard |
| `m` | Print performance metrics |

---

## 🧮 Scoring Algorithm

```
Score = HP×0.35 + Gaze×0.20 + EAR×0.20 + Blink×0.10 + PERCLOS×0.15

Where:
  HP      = cos((min(angle, 45°)/45°) × π/2)     Head pose score
  Gaze    = {center:1.0, up:0.7, left/right:0.5}  Gaze lookup
  EAR     = clip((ear - 0.15) / 0.11)             Eye openness
  Blink   = 1.0 if 10-22/min else decay           Blink rate
  PERCLOS = 1.0 - clip((perclos - 0.05) / 0.20)   Eye closure %
```

---

## 🏷️ State Classification

| State | Condition | Color |
|-------|-----------|-------|
| 🟢 Attentive | Score ≥ 0.75 | Green |
| 🟡 Distracted | 0.45 ≤ Score < 0.75 | Yellow |
| 🔴 Sleepy | Score < 0.45 + drowsy signals | Red |
| 🔵 Looking Away | Score < 0.45 + head turned | Blue |

---

## 🐳 Deployment

```bash
# Docker (simplest)
docker-compose up -d

# Direct
python main_app.py --camera 0 --port 8000

# Headless server
python main_app.py --no-display --log --port 8000
```

See `docs/DEPLOYMENT.md` for full guide (Docker, systemd, nginx).

---

## 🧪 Testing

```bash
pytest tests/ -v                    # All 130+ tests
pytest tests/ -v --cov=.            # With coverage
make test                           # Using Makefile
```

---

## ⚡ Performance

| Students | Resolution | Expected FPS | Hardware |
|----------|-----------|-------------|----------|
| 1-3 | 640px | 25-30 | Any modern laptop |
| 5-10 | 640px | 15-20 | i5/Ryzen 5+ |
| 10-15 | 480px | 10-15 | i7/Ryzen 7+ |

Press `m` during runtime to see per-stage timing breakdown.

---

## 📋 Configuration Priority

```
1. CLI args        → python main_app.py --camera 1
2. Env variables   → export CAMERA_INDEX=1
3. .env file       → CAMERA_INDEX=1
4. Code defaults   → camera_index=0
```
