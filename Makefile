# Makefile for Vietnamese Banking Stock Predictor

.PHONY: help setup install install-dev clean test lint format run-app run-pipeline view-data

# Default target
help:
	@echo "Vietnamese Banking Stock Predictor"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@echo "  setup          - Setup project directories and check requirements"
	@echo "  install        - Install production requirements"
	@echo "  install-dev    - Install development requirements"
	@echo "  clean          - Clean generated files and directories"
	@echo "  clean-data     - Clean data, models, and outputs"
	@echo "  clean-all      - Clean everything"
	@echo "  test           - Run tests"
	@echo "  lint           - Run code linting"
	@echo "  format         - Format code with black and isort"
	@echo "  run-app        - Start Streamlit web application"
	@echo "  run-pipeline   - Run full data pipeline"
	@echo "  collect-data   - Collect stock data"
	@echo "  engineer-features - Engineer features"
	@echo "  train-models   - Train all models"
	@echo "  view-data      - View database contents"
	@echo "  evaluate-models   - Analyze training results"
	@echo "  analyze-results   - Detailed results analysis"
	@echo "  quality-check  - Check data quality"
	@echo "  check-data        - Check collected banking data"

# Setup project
setup:
	python setup.py

# Install requirements
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# Clean generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -f *.log
	rm -f *.tmp
	rm -f *.temp
	rm -f profile.stats

clean-data:
	rm -rf data/processed/*
	rm -rf data/database/*
	rm -rf models/*
	rm -rf outputs/*
	rm -rf logs/*

clean-all: clean clean-data

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile=black

# Run applications
run-app:
	python run_app.py

run-app-direct:
	streamlit run app.py

# Pipeline commands
run-pipeline:
	python main.py full --models all

collect-data:
	python main.py collect

check-data:
	python check_data.py

engineer-features:
	python main.py features

train-models:
	python main.py train --models all

# Data viewing
view-data:
	python scripts/view_data.py tables

view-ticker:
	@read -p "Enter ticker symbol: " ticker; \
	python scripts/view_data.py ticker --name $$ticker

quality-check:
	python scripts/view_data.py quality



# Development helpers
jupyter:
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

profile:
	python -m cProfile -o profile.stats main.py collect
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Docker commands (if using Docker)
docker-build:
	docker build -t stock-predictor .

docker-run:
	docker run -p 8501:8501 -v $(PWD)/data:/app/data stock-predictor

# Backup and restore
backup-data:
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data/ models/ outputs/

restore-data:
	@read -p "Enter backup file name: " backup; \
	tar -xzf $$backup

# Model evaluation
evaluate-models:
	python scripts/manual_analysis.py

analyze-results:
	python scripts/analyze_training_results.py

# Update requirements
freeze-requirements:
	pip freeze > requirements-frozen.txt

# Git helpers
git-status:
	git status --porcelain

git-clean:
	git clean -fd
	git reset --hard HEAD