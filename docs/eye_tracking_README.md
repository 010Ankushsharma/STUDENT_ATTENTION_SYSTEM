# 👁️ Eye Tracking & Blink Detection Module

## Architecture

```
MediaPipe Face Mesh (478 landmarks)
         │
         ├──► EARCalculator ──────► Eye Aspect Ratio (L, R, Avg)
         │         │
         │         ▼
         │    BlinkDetector ──────► Blink count, rate, PERCLOS, duration
         │         │
         │         ▼
         │    DrowsinessDetector ─► Alert / Mild / Moderate / Severe
         │
         ├──► GazeEstimator ─────► Gaze direction (iris-based)
         │
         └──► EyeVisualizer ─────► Contours, iris, EAR bar, metrics panel
```

## Metrics Computed

| Metric | Description | Use |
|--------|-------------|-----|
| **EAR** | Eye Aspect Ratio (0-0.4) | Core eye-open indicator |
| **Blink Count** | Total blinks detected | Activity tracking |
| **Blink Rate** | Blinks/minute (sliding 60s) | Fatigue indicator |
| **PERCLOS** | % time eyes closed | Standard drowsiness measure |
| **Blink Duration** | Avg ms per blink | Fatigue progression |
| **Gaze Direction** | center/left/right/up/down | Attention direction |
| **Drowsiness Level** | alert → mild → moderate → severe | 4-indicator fusion |

## Quick Start

```bash
pip install -r requirements.txt
python run_eye_tracking.py
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `r` | Reset all detectors |
| `c` | Toggle eye contour outlines |
| `i` | Toggle iris markers |
| `m` | Toggle metrics panel |