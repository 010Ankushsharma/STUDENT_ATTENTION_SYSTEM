
"""
run_head_pose.py — Quick launcher with console alerts.
Run: python run_head_pose.py
"""
from head_pose import HeadPosePipeline


def main():
    print()
    print("  ╔════════════════════════════════════════════════════════╗")
    print("  ║  Head Pose Estimation — Live Demo                      ║")
    print("  ╠════════════════════════════════════════════════════════╣")
    print("  ║  Keys:                                                 ║")
    print("  ║    q / ESC  → Quit                                     ║")
    print("  ║    a        → Toggle 3D axes (X=red Y=green Z=blue)    ║")
    print("  ║    d        → Toggle direction labels                  ║")
    print("  ║    g        → Toggle yaw/pitch/roll gauges             ║")
    print("  ║    w        → Toggle face wireframe                    ║")
    print("  ║    p        → Toggle raw landmark points               ║")
    print("  ║    r        → Reset all Kalman filters                 ║")
    print("  ╚════════════════════════════════════════════════════════╝")
    print()

    pipeline = HeadPosePipeline(camera_index=0)

    def on_result(results, frame):
        for r in results:
            if not r.attention.is_attentive:
                print(f"  ⚠ Face {r.face_index}: {r.direction.combined_label.upper()} "
                      f"— {r.attention.zone} "
                      f"(yaw={r.pose.yaw}°, pitch={r.pose.pitch}°, "
                      f"score={r.attention.attention_score:.0%})")

    pipeline.run(on_result=on_result)


if __name__ == "__main__":
    main()