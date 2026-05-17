

"""
run_eye_tracking.py — Quick launcher with callback demo.
Run: python run_eye_tracking.py
"""
from eye_tracking import EyeTrackingPipeline


def main():
    print()
    print("  ╔═══════════════════════════════════════════════════╗")
    print("  ║  Eye Tracking & Blink Detection — Live Demo       ║")
    print("  ╠═══════════════════════════════════════════════════╣")
    print("  ║  Keys:                                            ║")
    print("  ║    q / ESC  → Quit                                ║")
    print("  ║    r        → Reset all detectors                 ║")
    print("  ║    c        → Toggle eye contour outlines         ║")
    print("  ║    i        → Toggle iris markers                 ║")
    print("  ║    m        → Toggle metrics panel                ║")
    print("  ╚═══════════════════════════════════════════════════╝")
    print()

    pipeline = EyeTrackingPipeline(camera_index=0)

    def on_result(results, frame):
        """Callback — logs drowsiness alerts to console."""
        for r in results:
            d = r.drowsiness
            if d.level.value != "alert":
                triggers = ", ".join(d.triggers)
                print(f"  ⚠ Face {r.face_index}: {d.level.value.upper()} "
                      f"(score={d.score}) — {triggers}")

    pipeline.run(on_result=on_result)


if __name__ == "__main__":
    main()

