
"""
core/errors.py
Custom exception hierarchy for the Student Attention System.

Exception Tree:
    AttentionSystemError
    ├── CameraError          Camera init/read failures
    ├── ProcessingError      Frame processing failures
    │   ├── FaceDetectionError
    │   ├── EyeTrackingError
    │   └── HeadPoseError
    ├── DatabaseError        DB connection/query failures
    ├── DashboardError       Web server errors
    └── ConfigError          Invalid configuration
"""


class AttentionSystemError(Exception):
    """Base exception for the entire system."""

    def __init__(self, message: str, module: str = "system", details: str = ""):
        self.module = module
        self.details = details
        super().__init__(f"[{module}] {message}")


class CameraError(AttentionSystemError):
    """Camera initialization or frame capture failure."""

    def __init__(self, message: str, camera_index: int = 0):
        self.camera_index = camera_index
        super().__init__(message, module="camera",
                        details=f"Camera index: {camera_index}")


class ProcessingError(AttentionSystemError):
    """Frame processing pipeline error."""

    def __init__(self, message: str, module: str = "processing",
                 frame_num: int = 0):
        self.frame_num = frame_num
        super().__init__(message, module=module,
                        details=f"Frame: {frame_num}")


class FaceDetectionError(ProcessingError):
    def __init__(self, message: str, frame_num: int = 0):
        super().__init__(message, module="face_detection", frame_num=frame_num)


class EyeTrackingError(ProcessingError):
    def __init__(self, message: str, frame_num: int = 0):
        super().__init__(message, module="eye_tracking", frame_num=frame_num)


class HeadPoseError(ProcessingError):
    def __init__(self, message: str, frame_num: int = 0):
        super().__init__(message, module="head_pose", frame_num=frame_num)


class DatabaseError(AttentionSystemError):
    """Database operation failure."""

    def __init__(self, message: str, operation: str = ""):
        self.operation = operation
        super().__init__(message, module="database",
                        details=f"Operation: {operation}")


class DashboardError(AttentionSystemError):
    """Web dashboard error."""

    def __init__(self, message: str):
        super().__init__(message, module="dashboard")


class ConfigError(AttentionSystemError):
    """Configuration validation error."""

    def __init__(self, message: str, parameter: str = ""):
        self.parameter = parameter
        super().__init__(message, module="config",
                        details=f"Parameter: {parameter}")

