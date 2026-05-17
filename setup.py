
"""
setup.py
Makes the Student Attention Detection System pip-installable.

Usage:
    pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="student-attention-system",
    version="1.0.0",
    description="AI-powered real-time classroom attention monitoring system",
    author="Student Attention Team",
    python_requires=">=3.9",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "dashboard": ["templates/*.html"],
    },
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "websockets>=12.0",
        "openpyxl>=3.1.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "httpx>=0.25.0", "pytest-cov>=4.1.0"],
    },
    entry_points={
        "console_scripts": [
            "attention-system=main_app:main",
            "attention-dashboard=dashboard.run_dashboard:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)