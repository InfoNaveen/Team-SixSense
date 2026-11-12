#!/usr/bin/env bash
# ==============================================================================
# CodeSpark 2026 — DISCOM Hackathon End-to-End Pipeline Runner
# ==============================================================================
set -e

echo "=== DISCOM Load Forecasting & Procurement Copilot Pipeline ==="

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "[ERROR] python command not found. Please install Python 3.10+."
    exit 1
fi

# Step 0: Ensure dependencies are installed
echo "--> Checking requirements..."
pip install -r requirements.txt -q

# Step 1: Check contracts
echo "--> Running contract verification..."
python scripts/check_contracts.py || true

# Step 2: Run master pipeline
# Pass --dummy if teammate files are not ready
if [ "$1" == "--dummy" ]; then
    echo "--> Running pipeline with dummy generator..."
    python run_all.py --dummy
else
    echo "--> Running pipeline..."
    python run_all.py
fi

# Step 3: Run unit tests
echo "--> Running procurement unit tests..."
python procurement/test_procurement.py

echo "=== Pipeline finished successfully ==="
echo "To launch the dashboard:"
echo "    streamlit run app/app.py"

