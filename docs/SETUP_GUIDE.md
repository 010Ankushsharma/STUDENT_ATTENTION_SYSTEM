# 🚀 Setup Guide — Student Attention Detection System

## Prerequisites

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.9+ | Runtime |
| pip | 21+ | Package manager |
| Webcam | USB/Built-in | Video input |
| OS | Windows 10+ / Ubuntu 20+ / macOS 12+ | Platform |

## Step 1: Clone & Setup

```bash
# Download and extract the project
unzip student_attention_system_final.zip
cd student_attention_system

# Create virtual environment (recommended)
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

## Step 2: Install Dependencies

```bash
# Standard install
pip install -r requirements.txt

# OR install as package (editable mode)
pip install -e .

# For development (includes test tools)
pip install -e ".[dev]"
```

## Step 3: Configure (Optional)

```bash
# Copy environment template
cp .env.example .env

# Edit .env to customize:
# - Camera index (if multiple cameras)
# - Dashboard port
# - Database path
# - Logging settings
```

## Step 4: Run

```bash
# Option A: Quick run (default camera + dashboard)
python run_system.py

# Option B: With specific options
python main_app.py --camera 0 --port 8000

# Option C: Using make (if available)
make run
```

## Step 5: Access Dashboard

Open your browser:
```
http://localhost:8000
```

Or press `w` in the OpenCV window to auto-open.

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t student-attention .

# Run with webcam access
docker run -d \
  --name attention-system \
  -p 8000:8000 \
  --device /dev/video0 \
  -v $(pwd)/data:/app/data \
  student-attention

# Or use docker-compose
docker-compose up -d
```

---

## 🔧 Troubleshooting

### Camera not detected
```bash
# List cameras (Linux)
v4l2-ctl --list-devices

# Test camera
python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"
```

### MediaPipe install fails
```bash
# Try specific version
pip install mediapipe==0.10.9

# On ARM/Mac M1:
pip install mediapipe-silicon  # (if available)
```

### Dashboard not loading
```bash
# Check if port is free
lsof -i :8000

# Try different port
python main_app.py --port 8080
```

### Low FPS
```bash
# Reduce processing resolution
python main_app.py --width 480

# Skip frames
python main_app.py --skip-frames 1

# Disable dashboard
python main_app.py --no-dashboard
```

---

## 📁 Data Locations

| Path | Content |
|------|---------|
| `attention_system.db` | SQLite database |
| `logs/` | Rotating log files |
| `exports/` | CSV/Excel reports |
| `screenshots/` | Saved screenshots |

---

## 🧪 Verify Installation

```bash
# Run all tests
pytest tests/ -v

# Quick syntax check
make lint

# Check system info
make info
```