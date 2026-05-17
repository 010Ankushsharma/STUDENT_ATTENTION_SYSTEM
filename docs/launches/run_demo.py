"""
run_demo.py — Quick launcher for the face detection demo.
Run: python run_demo.py
"""
from face_detection import FaceDetectionPipeline


def main():
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║   Student Face Detection — Live Demo         ║")
    print("  ╠══════════════════════════════════════════════╣")
    print("  ║  Keys:                                       ║")
    print("  ║    q / ESC  →  Quit                          ║")
    print("  ║    s        →  Save screenshot               ║")
    print("  ║    l        →  Toggle face landmarks         ║")
    print("  ║    f        →  Toggle FPS display            ║")
    print("  ║    r        →  Reset student IDs             ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    pipeline = FaceDetectionPipeline(camera_index=0)

    def on_frame(frame, students):
        """Optional callback — prints when student count changes."""
        active = [s for s in students.values() if s.disappeared == 0]
        # Uncomment below for per-frame logging:
        # for s in active:
        #     print(f"  Student {s.student_id}: {s.bbox}")

    pipeline.run(on_frame=on_frame)


if __name__ == "__main__":
    main()

