
"""
core/
Shared utilities, logging, error handling, and performance tools.
"""
from .logger import get_logger, setup_logging
from .errors import (
    AttentionSystemError,
    CameraError,
    ProcessingError,
    DatabaseError,
    ConfigError,
)
from .performance import PerformanceMonitor, FPSCounter, Timer
from .config_loader import load_config_from_env

__all__ = [
    "get_logger", "setup_logging",
    "AttentionSystemError", "CameraError", "ProcessingError",
    "DatabaseError", "ConfigError",
    "PerformanceMonitor", "FPSCounter", "Timer",
    "load_config_from_env",
]

