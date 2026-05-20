# 👁️ Face Detection Module — Student Attention System

## Architecture

```
Raw Frame
    │
    ▼
┌──────────────────┐
│  Preprocessor    │   Resize + CLAHE (lighting fix)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  FaceDetector    │   MediaPipe Face Mesh (468 landmarks)
│  (detector.py)   │   or Face Detection (bbox only)
└────────┬─────────┘
         │  List[DetectedFace]
         ▼
┌──────────────────┐
│ CentroidTracker  │   Hungarian matching → persistent IDs
│  (tracker.py)    │   handles appear/disappear/reappear
└────────┬─────────┘
         │  Dict[id → TrackedStudent]
         ▼
┌──────────────────┐
│  FrameAnnotator  │   Rounded boxes, IDs, confidence, FPS
│  (annotator.py)  │
└────────┬─────────┘
         │
         ▼
    Annotated Frame
```

## Quick Start

```bash
pip install -r requirements.txt
python run_demo.py
```

## Module Usage

```python
from face_detection import FaceDetectionPipeline

# Option 1: Live webcam loop (blocking)
pipeline = FaceDetectionPipeline(camera_index=0)
pipeline.run()

# Option 2: Process single frames (non-blocking)
pipeline = FaceDetectionPipeline()
annotated_frame, students = pipeline.process_frame(my_frame)

for sid, student in students.items():
    print(f"Student {sid} at {student.bbox}")
```

## File Structure

```
face_detection/
├── __init__.py             # Public API exports
├── __main__.py             # python -m face_detection
├── config.py               # All tunable parameters
├── preprocessor.py         # CLAHE + resize
├── detector.py             # MediaPipe face detection
├── tracker.py              # Centroid-based ID tracking
├── annotator.py            # Drawing utilities
├── pipeline.py             # End-to-end orchestrator
└── integration_example.py  # FastAPI integration guide
```

## Performance Optimization Tips

| Technique | Impact | How |
|-----------|--------|-----|
| Frame skip | 2× faster | `cfg.skip_frames = 1` (process every 2nd) |
| Lower resolution | 3× faster | `cfg.process_width = 480` |
| Face Detection mode | 1.5× faster | `use_mesh=False` (loses landmarks) |
| Disable CLAHE | 10% faster | `cfg.apply_clahe = False` |
| Reduce max faces | Variable | `cfg.max_num_faces = 5` |
| JPEG quality | Less bandwidth | `cfg.jpeg_quality = 60` |
| GPU MediaPipe | 3× faster | Requires special MediaPipe build |

## Detection Modes

| Mode | Landmarks | Speed | Best For |
|------|-----------|-------|----------|
| Face Mesh | 468 points | ~25 FPS | Full system (EAR + head pose) |
| Face Detection | bbox only | ~40 FPS | Quick face counting |

## Keyboard Controls (Live Mode)

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `s` | Save screenshot |
| `l` | Toggle landmark dots |
| `f` | Toggle FPS counter |
| `r` | Reset all student IDs |