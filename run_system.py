"""
run_system.py
═══════════════════════════════════════════════════
  Quick Launcher — Student Attention Detection System
═══════════════════════════════════════════════════

Usage:
    python run_system.py

This is the simplest way to start the system with
default settings (camera 0, dashboard on port 8000).

For more options:
    python main_app.py --help
"""
import sys

from core import setup_logging
from main_app import StudentAttentionSystem, main


if __name__ == "__main__":
    # If CLI args provided, use full argument parser
    if len(sys.argv) > 1:
        main()
    else:
        # Default: camera 0, dashboard on, show video
        setup_logging(level="INFO", log_to_file=True)
        system = StudentAttentionSystem(
            camera_index=0,
            show_video=True,
            enable_dashboard=True,
        )
        system.run()
