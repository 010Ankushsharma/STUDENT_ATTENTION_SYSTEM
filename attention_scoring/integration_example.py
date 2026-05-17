"""
attention_scoring/integration_example.py
Shows how to integrate the scoring pipeline with the other modules.
"""
from attention_scoring import ScoringPipeline, Alert


def example_basic_usage():
    """Basic usage with manual signal input."""
    pipeline = ScoringPipeline()

    # Register alert callback
    def on_alert(alert: Alert):
        print(f"  🚨 ALERT: {alert.message} (severity={alert.severity.value})")

    pipeline.set_alert_callback(on_alert)

    # Simulate 100 frames of an attentive student
    print("--- Attentive Student ---")
    for frame in range(100):
        result = pipeline.update(
            student_id=0,
            ear=0.29,
            gaze_direction="center",
            yaw=3.0,
            pitch=-2.0,
            blink_rate=16.0,
            perclos=0.03,
            drowsiness_level="alert",
            head_direction="forward",
        )
    print(f"  Score: {result.score:.2f}, State: {result.state}, Attention: {result.attention_pct}%")

    # Simulate distracted student (looking right)
    print("\n--- Distracted Student ---")
    for frame in range(100):
        result = pipeline.update(
            student_id=1,
            ear=0.27,
            gaze_direction="right",
            yaw=25.0,
            pitch=5.0,
            blink_rate=18.0,
            perclos=0.06,
            drowsiness_level="alert",
            head_direction="slight_right",
        )
    print(f"  Score: {result.score:.2f}, State: {result.state}, Attention: {result.attention_pct}%")

    # Simulate sleepy student
    print("\n--- Sleepy Student ---")
    for frame in range(100):
        result = pipeline.update(
            student_id=2,
            ear=0.16,
            gaze_direction="down",
            yaw=-5.0,
            pitch=-15.0,
            blink_rate=28.0,
            perclos=0.22,
            drowsiness_level="moderate_drowsy",
            head_direction="slight_down",
        )
    print(f"  Score: {result.score:.2f}, State: {result.state}, Attention: {result.attention_pct}%")

    # Simulate looking away student
    print("\n--- Looking Away Student ---")
    for frame in range(100):
        result = pipeline.update(
            student_id=3,
            ear=0.30,
            gaze_direction="left",
            yaw=-40.0,
            pitch=0.0,
            blink_rate=14.0,
            perclos=0.04,
            drowsiness_level="alert",
            head_direction="left",
        )
    print(f"  Score: {result.score:.2f}, State: {result.state}, Attention: {result.attention_pct}%")

    # Class summary
    print("\n--- Class Summary ---")
    summary = pipeline.get_class_summary()
    print(f"  Total students: {summary['total']}")
    print(f"  Attentive: {summary['attentive']}, Distracted: {summary['distracted']}")
    print(f"  Sleepy: {summary['sleepy']}, Looking away: {summary['looking_away']}")
    print(f"  Class average score: {summary['avg_score']:.2f}")

    # Leaderboard
    print("\n--- Leaderboard ---")
    for s in pipeline.get_leaderboard():
        print(f"  Student {s['student_id']}: {s['attention_pct']:.0f}% — {s['state']}")

    # Alerts
    print(f"\n--- Alerts ({len(pipeline.get_alerts())}) ---")
    for a in pipeline.get_alerts():
        print(f"  [{a.severity.value}] {a.message}")


def example_full_integration():
    """
    Integration with eye_tracking, head_pose, and face_detection modules.

    This is how you'd wire everything together in the main app.
    """
    # Pseudocode — requires the other modules to be installed:
    #
    # from face_detection import FaceDetectionPipeline
    # from eye_tracking import EyeTrackingPipeline
    # from head_pose import HeadPosePipeline
    # from attention_scoring import ScoringPipeline
    #
    # face_pipeline = FaceDetectionPipeline()
    # eye_pipeline = EyeTrackingPipeline()
    # head_pipeline = HeadPosePipeline()
    # scoring = ScoringPipeline()
    #
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #
    #     # Detect faces
    #     frame, students = face_pipeline.process_frame(frame)
    #
    #     # For each student face:
    #     for student in students.values():
    #         # Eye tracking
    #         eye_results, _ = eye_pipeline.process_frame(frame)
    #         eye = eye_results[0] if eye_results else None
    #
    #         # Head pose
    #         pose_results, _ = head_pipeline.process_frame(frame)
    #         pose = pose_results[0] if pose_results else None
    #
    #         # Score attention
    #         result = scoring.update(
    #             student_id=student.student_id,
    #             ear=eye.ear_avg if eye else 0.3,
    #             gaze_direction=eye.gaze.direction_label if eye else "center",
    #             yaw=pose.pose.yaw if pose else 0,
    #             pitch=pose.pose.pitch if pose else 0,
    #             blink_rate=eye.blink_metrics.blink_rate_per_min if eye else 15,
    #             perclos=eye.blink_metrics.perclos if eye else 0.05,
    #             drowsiness_level=eye.drowsiness.level.value if eye else "alert",
    #             head_direction=pose.direction.combined_label if pose else "forward",
    #         )
    #
    #         # Use result
    #         print(f"Student {student.student_id}: {result.state} ({result.score_pct}%)")
    pass


if __name__ == "__main__":
    example_basic_usage()