#!/bin/bash

# Simple script to run all tests in the repository
# Usage: ./infra/scripts/run_tests.sh

# Set PYTHONPATH to include the source directory
export PYTHONPATH=$(pwd)/source

# Run pytest
echo "Running pytest..."
pytest

# Optional: Run simulation demos to verify they work
echo -e "\nRunning Inference Simulator demo..."
python source/vlm_inference_lab/simulation/inference_simulator.py

echo -e "\nRunning VLM Embedding demo..."
python source/vlm_inference_lab/embeddings/demo.py
