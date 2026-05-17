
# ============================================
# Student Attention Detection System
# Docker Deployment
# ============================================
FROM python:3.11-slim

# System dependencies for OpenCV & MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p /app/data /app/exports /app/logs

# Expose dashboard port
EXPOSE 8000

# Environment
ENV PYTHONUNBUFFERED=1
ENV ATTENTION_DB_PATH=/app/data/attention_system.db
ENV ATTENTION_LOG_DIR=/app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Default: run with dashboard
CMD ["python", "main_app.py", "--camera", "0", "--dashboard"]

