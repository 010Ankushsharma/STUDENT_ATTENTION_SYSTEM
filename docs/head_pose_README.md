# 🧭 Head Pose Estimation Module

## Architecture

```
MediaPipe Face Mesh (478 landmarks)
         │
         ▼
┌─────────────────────┐
│   LandmarkMapper    │  Extract 6 or 14 key facial points
│ (landmark_mapper.py)│  Map to 3D anthropometric model
└─────────┬───────────┘
          │  MappedLandmarks (2D + 3D pairs)
          ▼
┌─────────────────────┐
│   PoseCalculator    │  cv2.solvePnP → rotation vector
│ (pose_calculator.py)│  Rodrigues → rotation matrix
│                     │  RQDecomp3x3 → Euler angles
│                     │  + Kalman smoothing
└─────────┬───────────┘
          │  PoseResult (yaw, pitch, roll)
          ├───────────────────────────┐
          ▼                           ▼
┌───────────────────┐     ┌────────────────────┐
│DirectionClassifier│     │  AttentionJudge     │
│ forward/left/     │     │  attentive/marginal/│
│ right/up/down     │     │  inattentive        │
└─────────┬─────────┘     └──────────┬─────────┘
          │                          │
          └────────────┬─────────────┘
                       ▼
              ┌─────────────────┐
              │  PoseVisualizer │  3D axes, gauges,
              │                 │  labels, wireframe
              └─────────────────┘
```

## Angle Conventions

```
        YAW (horizontal)              PITCH (vertical)
                                      
  LEFT ◄────── 0° ──────► RIGHT    DOWN ◄────── 0° ──────► UP
       -60°    ↑    +60°              -40°    ↑    +40°
               │                              │
           FORWARD                        FORWARD
```

## Direction Classification Grid

```
                 PITCH
     UP-LEFT      UP      UP-RIGHT
        ╲         ↑         ╱
  LEFT ── SLIGHT ─ FORWARD ─ SLIGHT ── RIGHT  → YAW
        ╱         ↓         ╲
   DOWN-LEFT    DOWN    DOWN-RIGHT
```

## Quick Start

```bash
pip install -r requirements.txt
python run_head_pose.py
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `a` | Toggle 3D axes (X=red, Y=green, Z=blue) |
| `d` | Toggle direction labels |
| `g` | Toggle yaw/pitch/roll gauge arcs |
| `w` | Toggle face wireframe |
| `p` | Toggle raw landmark points |
| `r` | Reset Kalman filters & trackers |

## Integration

```python
from head_pose import HeadPosePipeline

pipeline = HeadPosePipeline()
results, annotated = pipeline.process_frame(bgr_frame)

for r in results:
    print(f"Yaw: {r.pose.yaw}°, Pitch: {r.pose.pitch}°")
    print(f"Direction: {r.direction.combined_label}")
    print(f"Attention: {r.attention.zone} ({r.attention.attention_score:.0%})")
```