# 🚢 Deployment Guide — Student Attention Detection System

## Deployment Options

| Method | Best For | Complexity |
|--------|----------|------------|
| Direct Python | Development, single machine | ⭐ |
| Docker | Isolated deployment, CI/CD | ⭐⭐ |
| Docker Compose | Multi-service setups | ⭐⭐ |
| Systemd Service | Linux server, auto-start | ⭐⭐⭐ |

---

## Option 1: Direct Python (Simplest)

```bash
# Install
pip install -r requirements.txt

# Run
python main_app.py --camera 0 --port 8000

# Access dashboard: http://<server-ip>:8000
```

---

## Option 2: Docker

### Build
```bash
docker build -t student-attention:latest .
```

### Run
```bash
docker run -d \
  --name attention-system \
  --restart unless-stopped \
  -p 8000:8000 \
  --device /dev/video0:/dev/video0 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/exports:/app/exports \
  -e CAMERA_INDEX=0 \
  -e DASHBOARD_PORT=8000 \
  student-attention:latest
```

### Manage
```bash
docker logs -f attention-system     # View logs
docker exec -it attention-system bash  # Shell access
docker stop attention-system        # Stop
docker start attention-system       # Start
```

---

## Option 3: Docker Compose

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

---

## Option 4: Linux Systemd Service

### Create service file
```bash
sudo nano /etc/systemd/system/attention-system.service
```

```ini
[Unit]
Description=Student Attention Detection System
After=network.target

[Service]
Type=simple
User=attention
WorkingDirectory=/opt/attention-system
ExecStart=/opt/attention-system/venv/bin/python main_app.py --no-display --log
Restart=always
RestartSec=10
Environment=DASHBOARD_PORT=8000
Environment=DB_PATH=/opt/attention-system/data/attention.db
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Enable & Start
```bash
sudo systemctl daemon-reload
sudo systemctl enable attention-system
sudo systemctl start attention-system
sudo systemctl status attention-system
```

---

## Network Configuration

### Firewall (Allow dashboard access)
```bash
# Ubuntu/Debian
sudo ufw allow 8000/tcp

# CentOS/RHEL
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

### Reverse Proxy (Nginx)
```nginx
server {
    listen 80;
    server_name attention.school.local;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    location /video_feed {
        proxy_pass http://localhost:8000/video_feed;
        proxy_buffering off;
    }
}
```

---

## Performance Tuning

### For low-end hardware (Raspberry Pi, old laptops):
```bash
python main_app.py --width 320 --skip-frames 2 --no-dashboard
```

### For high-end hardware (GPU workstation):
```bash
python main_app.py --width 960 --camera 0
```

### Environment variables for tuning:
```env
PROCESS_WIDTH=480        # Lower = faster
SKIP_FRAMES=1            # Skip every other frame
TARGET_FPS=15            # Lower target
LOG_EVERY_N_FRAMES=10    # Less DB writes
BATCH_SIZE=100           # Larger batches
```

---

## Monitoring & Maintenance

### Check system health:
```bash
curl http://localhost:8000/api/status
```

### Database maintenance:
```python
from database import DatabaseManager
db = DatabaseManager()
db.cleanup_old_data(days=30)  # Remove old data
db.vacuum()                    # Reclaim space
print(f"DB size: {db.get_db_size_mb():.2f} MB")
```

### Log rotation:
Logs auto-rotate at 10MB (5 backups kept). Check `logs/` directory.

---

## Production Checklist

- [ ] Camera accessible and tested
- [ ] Dependencies installed
- [ ] .env configured
- [ ] Dashboard accessible from network
- [ ] Firewall rules set
- [ ] Log rotation working
- [ ] DB backups scheduled
- [ ] Auto-restart on crash (systemd/Docker)
- [ ] Monitoring endpoint (`/api/status`) accessible
- [ ] Sufficient disk space for DB growth (~1MB/hour typical)