# ==============================================================================
# CodeSpark 2026 — DISCOM Hackathon Makefile
# ==============================================================================

.PHONY: help setup dummy check test run app clean

help:
	@echo "Available commands:"
	@echo "  make setup   - Install project dependencies"
	@echo "  make dummy   - Generate contract dummy forecasts for testing"
	@echo "  make check   - Validate contract schemas across all modules"
	@echo "  make test    - Run procurement unit tests"
	@echo "  make run     - Execute full pipeline end-to-end"
	@echo "  make app     - Launch Streamlit dashboard"
	@echo "  make clean   - Remove cache files"

setup:
	pip install -r requirements.txt

dummy:
	python stubs/generate_dummy_forecasts.py

check:
	python scripts/check_contracts.py

test:
	python procurement/test_procurement.py

run:
	python run_all.py

app:
	streamlit run app/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

