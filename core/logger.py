
"""
core/logger.py
Centralized logging with file rotation, coloured console output,
and per-module loggers.

Usage:
    from core import get_logger
    logger = get_logger("face_detection")
    logger.info("Detected 3 faces")
    logger.warning("Low confidence detection")
    logger.error("Camera disconnected")
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional


# ── Colour codes for console ──
class _Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


LEVEL_COLORS = {
    "DEBUG": _Colors.GRAY,
    "INFO": _Colors.GREEN,
    "WARNING": _Colors.YELLOW,
    "ERROR": _Colors.RED,
    "CRITICAL": _Colors.MAGENTA,
}


class ColoredFormatter(logging.Formatter):
    """Coloured console formatter with timestamps."""

    def format(self, record):
        color = LEVEL_COLORS.get(record.levelname, _Colors.RESET)
        # Format: [HH:MM:SS] MODULE  LEVEL  message
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        module = record.name.split(".")[-1][:15].ljust(15)
        level = record.levelname[:5].ljust(5)
        msg = record.getMessage()

        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        return (
            f"{_Colors.GRAY}[{timestamp}]{_Colors.RESET} "
            f"{_Colors.CYAN}{module}{_Colors.RESET} "
            f"{color}{level}{_Colors.RESET} {msg}"
        )


class FileFormatter(logging.Formatter):
    """Plain text formatter for log files."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# ── Global state ──
_initialized = False
_log_dir = "logs"


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
    max_file_size_mb: int = 10,
    backup_count: int = 5,
):
    """
    Initialize the logging system (call once at startup).

    Args:
        level:            Minimum log level (DEBUG/INFO/WARNING/ERROR)
        log_to_file:      Enable file logging with rotation
        log_dir:          Directory for log files
        max_file_size_mb: Max size per log file before rotation
        backup_count:     Number of rotated files to keep
    """
    global _initialized, _log_dir
    if _initialized:
        return

    _log_dir = log_dir
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColoredFormatter())
    console.setLevel(logging.DEBUG)
    root.addHandler(console)

    # File handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir,
            f"attention_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(FileFormatter())
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)

    _initialized = True

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("mediapipe").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.

    Usage:
        logger = get_logger("face_detection")
        logger.info("Ready")
    """
    if not _initialized:
        setup_logging()
    return logging.getLogger(f"attention.{name}")

