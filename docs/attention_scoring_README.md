# 🧠 AI Attention Scoring Module

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT SIGNALS                                 │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────┤
│  Head Pose   │    Gaze      │     EAR      │ Blink Rate   │ PERCLOS │
│  (yaw,pitch) │ (direction)  │ (eye open)   │ (per min)    │ (%)     │
└──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┴────┬────┘
       │              │              │              │            │
       ▼              ▼              ▼              ▼            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SIGNAL FUSION (signal_fusion.py)                  │
│  Normalise each signal (0-1) using smooth scoring functions          │
│  Apply configurable weights: HP=35%, Gaze=20%, EAR=20%,             │
│                               Blink=10%, PERCLOS=15%                 │
│  Output: raw_weighted_score (0.0 – 1.0)                             │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  ATTENTION SCORER (scorer.py)                         │
│  Exponential Moving Average smoothing                                │
│  Prevents jitter from frame-to-frame noise                           │
│  Output: smoothed_score (0.0 – 1.0)                                 │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│               STATE CLASSIFIER (state_classifier.py)                  │
│  score ≥ 0.75 → ATTENTIVE                                           │
│  score 0.45-0.74 → DISTRACTED                                       │
│  score < 0.45 + drowsy signals → SLEEPY                              │
│  score < 0.45 + head away → LOOKING_AWAY                             │
│  + Hysteresis (8-frame delay prevents flickering)                    │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  ALERT ENGINE (alert_engine.py)                       │
│  Triggers: LOW_ATTENTION, SLEEPY, LOOKING_AWAY, RAPID_DECLINE       │
│  Per-student cooldown (30s), max cap, severity levels               │
│  Callback support for external notifications                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Scoring Formula

```
Score = Σ (weight_i × signal_score_i)

Where:
  signal_score_i = f(raw_signal)  mapped to [0.0, 1.0]
  
  head_pose_score  = cosine_decay(|yaw|, 10°→45°) × cosine_decay(|pitch|, 8°→35°)
  gaze_score       = lookup_table(direction)
  ear_score        = linear(EAR, 0.15→0.26)
  blink_score      = optimal_range(rate, 10-22/min)
  perclos_score    = linear(perclos, 5%→25%)
```

## Quick Start

```bash
pip install -r requirements.txt

# Run integration example
python -m attention_scoring.integration_example

# Run tests
pytest tests/ -v
```

## Usage

```python
from attention_scoring import ScoringPipeline

pipeline = ScoringPipeline()

result = pipeline.update(
    student_id=0,
    ear=0.28,
    gaze_direction="center",
    yaw=5.0, pitch=3.0,
    blink_rate=16.0,
    perclos=0.04,
    drowsiness_level="alert",
    head_direction="forward",
)

print(result.score)          # 0.87
print(result.state)          # "attentive"
print(result.attention_pct)  # 92.3
print(result.alerts)         # []
```

## Configuration

All thresholds in `config.py`:

```python
# Signal weights (must sum to 1.0)
weights = {
    "head_pose": 0.35,
    "gaze":      0.20,
    "ear":       0.20,
    "blink_rate":0.10,
    "perclos":   0.15,
}

# State thresholds
threshold_attentive = 0.75    # Score above → attentive
threshold_distracted = 0.45   # Score above → distracted, below → sleepy/away

# Smoothing
smoothing_window = 15         # ~0.5s at 30fps
state_change_delay = 8        # Frames before confirming state change
score_momentum = 0.3          # EMA factor (higher = faster response)
```