
"""
face_detection/integration_example.py
Shows how to integrate the face detection module into FastAPI
or any other application.
"""
import cv2
import base64
import time
import threading
from face_detection import FaceDetectionPipeline


# ---- Example 1: Basic single-frame usage ----
def example_single_frame():
    """Process a single image file."""
    pipeline = FaceDetectionPipeline()

    frame = cv2.imread("classroom.jpg")
    if frame is None:
        print("Could not load image")
        return

    annotated, students = pipeline.process_frame(frame)

    print(f"Detected {len(students)} students:")
    for sid, s in students.items():
        print(f"  Student {sid}: bbox={s.bbox}, conf={s.confidence:.0%}")

    cv2.imwrite("output_detected.jpg", annotated)
    pipeline.detector.release()


# ---- Example 2: FastAPI WebSocket integration ----
def example_fastapi_integration():
    """
    Pseudocode for FastAPI integration.
    The pipeline runs in a background thread and pushes
    frames via a shared variable.
    """
    pipeline = FaceDetectionPipeline(camera_index=0)
    latest_data = {"frame_b64": "", "students": {}}
    lock = threading.Lock()

    def camera_thread():
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, students = pipeline.process_frame(frame)

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buf).decode("utf-8")

            with lock:
                latest_data["frame_b64"] = b64
                latest_data["students"] = pipeline.get_student_data()

            time.sleep(0.03)
        cap.release()

    # Start thread
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()

    # In your FastAPI WebSocket handler:
    # @app.websocket("/ws/live")
    # async def ws(websocket):
    #     while True:
    #         with lock:
    #             await websocket.send_json(latest_data)
    #         await asyncio.sleep(0.1)


# ---- Example 3: With callback for logging ----
def example_with_callback():
    """Run pipeline with a callback that fires each frame."""
    pipeline = FaceDetectionPipeline(camera_index=0)

    def on_frame(frame, students):
        active = sum(1 for s in students.values() if s.disappeared == 0)
        if active > 0:
            ids = [s.student_id for s in students.values() if s.disappeared == 0]
            print(f"  Frame: {active} students visible — IDs: {ids}")

    pipeline.run(on_frame=on_frame)


if __name__ == "__main__":
    example_with_callback()
