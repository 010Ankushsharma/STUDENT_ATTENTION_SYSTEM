# ============================================
# Student Attention Detection System
# Makefile for common tasks
# ============================================

.PHONY: install run dashboard test lint clean docker docs

# ── Setup ──
install:
	pip install -r requirements.txt
	pip install -e .
	@echo "✅ Installation complete"

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	@echo "✅ Dev installation complete"

# ── Run ──
run:
	python main_app.py --camera 0

run-nodisplay:
	python main_app.py --camera 0 --no-display --log

dashboard:
	python -m dashboard.run_dashboard --port 8000

# ── Testing ──
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=. --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

test-fast:
	pytest tests/ -x -q

# ── Docker ──
docker-build:
	docker build -t student-attention .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# ── Maintenance ──
clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache htmlcov
	rm -f *.db *.log

clean-all: clean
	rm -rf data/ exports/ logs/

lint:
	python -m py_compile main_app.py
	python -m py_compile main_config.py
	@echo "✅ No syntax errors"

# ── Info ──
info:
	@echo "Student Attention Detection System v1.0.0"
	@echo "Modules: face_detection, eye_tracking, head_pose,"
	@echo "         attention_scoring, database, dashboard"
	@echo ""
	@echo "Commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make run        - Run full system"
	@echo "  make dashboard  - Run dashboard only"
	@echo "  make test       - Run all tests"
	@echo "  make docker-run - Run in Docker"

