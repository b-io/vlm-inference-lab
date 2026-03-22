#!/bin/bash

# Simple script to run all tests in the repository
# Usage: ./infra/scripts/run_tests.sh

# Set PYTHONPATH to include the source directory
export PYTHONPATH=$(pwd)/source

# Run pytest
echo "--------------------------------------------------------"
echo "Running pytest..."
echo "--------------------------------------------------------"
pytest

# Optional: Run simulation demos to verify they work
echo -e "\n--------------------------------------------------------"
echo "Running Inference Simulator demo..."
echo "--------------------------------------------------------"
python source/vlm_inference_lab/simulation/inference_simulator.py

echo -e "\n--------------------------------------------------------"
echo "Running VLM Embedding demo..."
echo "--------------------------------------------------------"
python source/vlm_inference_lab/embeddings/demo.py
