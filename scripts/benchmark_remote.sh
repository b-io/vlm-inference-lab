#!/bin/bash
set -euo pipefail

# Remote Benchmark Helper Script
# Usage: ./benchmark_remote.sh <base_url> <model_id> [num_requests] [concurrency] [extra_args]

BASE_URL=${1:?Required base_url}
MODEL_ID=${2:?Required model_id}
NUM_REQUESTS=${3:-10}
CONCURRENCY=${4:-2}
shift 4 || true
EXTRA_ARGS="$*"

OUTPUT_DIR="results/remote"
mkdir -p "${OUTPUT_DIR}"

echo "--------------------------------------------------------"
echo "Starting Remote Benchmark"
echo "URL         : ${BASE_URL}"
echo "Model       : ${MODEL_ID}"
echo "Requests    : ${NUM_REQUESTS}"
echo "Concurrency : ${CONCURRENCY}"
echo "--------------------------------------------------------"

# Ensure dependencies are installed
export PYTHONPATH="source:${PYTHONPATH:-}"

# Run the benchmark
python3 -m vlm_inference_lab.experiments.benchmark_serving \
    --url "${BASE_URL}" \
    --model "${MODEL_ID}" \
    --num-requests "${NUM_REQUESTS}" \
    --concurrency "${CONCURRENCY}" \
    --output-dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}

echo "--------------------------------------------------------"
echo "Benchmark Complete"
echo "Results saved in: ${OUTPUT_DIR}"
echo "--------------------------------------------------------"
