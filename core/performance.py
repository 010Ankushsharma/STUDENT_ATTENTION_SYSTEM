
"""
core/performance.py
Performance monitoring, FPS counting, and profiling utilities.

Usage:
    from core import PerformanceMonitor, FPSCounter, Timer

    fps = FPSCounter()
    monitor = PerformanceMonitor()

    while True:
        with Timer("face_detection") as t:
            detect_faces(frame)
        monitor.record("face_detection", t.elapsed_ms)

        fps.tick()
        print(f"FPS: {fps.get():.1f}")

    monitor.print_report()
"""
import time
from collections import deque, defaultdict
from typing import Dict, Optional
from dataclasses import dataclass, field


class FPSCounter:
    """
    Accurate FPS counter using rolling window.

    Usage:
        fps = FPSCounter(window=30)
        # In loop:
        fps.tick()
        print(fps.get())
    """

    def __init__(self, window: int = 30):
        self._timestamps = deque(maxlen=window)
        self._last_fps = 0.0

    def tick(self):
        """Call once per frame."""
        self._timestamps.append(time.perf_counter())

    def get(self) -> float:
        """Get current FPS."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        fps = (len(self._timestamps) - 1) / elapsed
        self._last_fps = fps
        return fps

    def get_ms_per_frame(self) -> float:
        """Get milliseconds per frame."""
        fps = self.get()
        return (1000.0 / fps) if fps > 0 else 0.0


class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer("my_operation") as t:
            do_something()
        print(f"Took {t.elapsed_ms:.1f}ms")

    Or as decorator:
        @Timer.decorate("my_function")
        def my_function():
            ...
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000.0

    @staticmethod
    def decorate(name: str = ""):
        """Use as a decorator to time function calls."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000.0
                # Store on function for access
                wrapper.last_elapsed_ms = elapsed
                return result
            wrapper.last_elapsed_ms = 0.0
            return wrapper
        return decorator


@dataclass
class _StageStats:
    """Stats for a single pipeline stage."""
    name: str
    times: deque = field(default_factory=lambda: deque(maxlen=100))
    total_calls: int = 0
    total_time_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.times) if self.times else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.times) if self.times else 0.0


class PerformanceMonitor:
    """
    Tracks timing for all pipeline stages.

    Usage:
        monitor = PerformanceMonitor()

        # Record timings:
        monitor.record("face_detection", 12.5)
        monitor.record("eye_tracking", 8.3)
        monitor.record("head_pose", 6.1)
        monitor.record("scoring", 1.2)
        monitor.record("total_frame", 35.0)

        # Get report:
        report = monitor.get_report()
        monitor.print_report()
    """

    def __init__(self):
        self._stages: Dict[str, _StageStats] = {}
        self._start_time = time.time()

    def record(self, stage: str, elapsed_ms: float):
        """Record a timing measurement for a stage."""
        if stage not in self._stages:
            self._stages[stage] = _StageStats(name=stage)
        stats = self._stages[stage]
        stats.times.append(elapsed_ms)
        stats.total_calls += 1
        stats.total_time_ms += elapsed_ms

    def get_report(self) -> Dict[str, dict]:
        """Get full performance report."""
        report = {}
        for name, stats in self._stages.items():
            report[name] = {
                "avg_ms": round(stats.avg_ms, 2),
                "max_ms": round(stats.max_ms, 2),
                "min_ms": round(stats.min_ms, 2),
                "total_calls": stats.total_calls,
                "total_time_sec": round(stats.total_time_ms / 1000, 2),
            }
        return report

    def print_report(self):
        """Print formatted performance report."""
        uptime = time.time() - self._start_time
        print()
        print("  ╔══════════════════════════════════════════════════════╗")
        print("  ║          PERFORMANCE REPORT                          ║")
        print("  ╠══════════════════════════════════════════════════════╣")
        print(f"  ║  Uptime: {uptime:.1f}s")
        print("  ╠══════════════════════════════════════════════════════╣")
        print(f"  ║  {'Stage':<20} {'Avg(ms)':<10} {'Max(ms)':<10} {'Calls':<8} ║")
        print("  ╠══════════════════════════════════════════════════════╣")

        for name, stats in sorted(self._stages.items(),
                                   key=lambda x: x[1].avg_ms, reverse=True):
            print(f"  ║  {name:<20} {stats.avg_ms:<10.2f} "
                  f"{stats.max_ms:<10.2f} {stats.total_calls:<8} ║")

        print("  ╚══════════════════════════════════════════════════════╝")
        print()

    def get_total_frame_budget(self, target_fps: int = 30) -> dict:
        """Check if pipeline fits within frame budget."""
        budget_ms = 1000.0 / target_fps
        total_avg = sum(s.avg_ms for s in self._stages.values()
                       if s.name != "total_frame")
        return {
            "budget_ms": budget_ms,
            "used_ms": round(total_avg, 2),
            "remaining_ms": round(budget_ms - total_avg, 2),
            "utilization_pct": round(total_avg / budget_ms * 100, 1),
            "fits_budget": total_avg < budget_ms,
        }

    def reset(self):
        """Reset all stats."""
        self._stages.clear()
        self._start_time = time.time()