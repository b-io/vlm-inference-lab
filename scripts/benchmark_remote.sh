#!/bin/bash
set -euo pipefail

# Remote Benchmark Helper Script
# Usage: ./benchmark_remote.sh <base_url> <model_id> [num_requests] [concurrency] [extra_args]
#        ./benchmark_remote.sh <base_url> <model_id> --tier <smoke|latency|throughput> [extra_args]

BASE_URL=${1:?Required base_url}
MODEL_ID=${2:?Required model_id}
shift 2

NUM_REQUESTS=10
CONCURRENCY=2
TIER=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)
            TIER="$2"
            shift 2
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                if [ -z "${_REQ_SET:-}" ]; then
                    NUM_REQUESTS="$1"
                    _REQ_SET=1
                elif [ -z "${_CONC_SET:-}" ]; then
                    CONCURRENCY="$1"
                    _CONC_SET=1
                else
                    EXTRA_ARGS+=("$1")
                fi
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

OUTPUT_DIR="results/remote"
mkdir -p "${OUTPUT_DIR}"

echo "--------------------------------------------------------"
echo "Starting Remote Benchmark"
echo "URL         : ${BASE_URL}"
echo "Model       : ${MODEL_ID}"
if [ -n "$TIER" ]; then echo "Tier        : ${TIER}"; fi
echo "Requests    : ${NUM_REQUESTS} (default if not tier/arg)"
echo "Concurrency : ${CONCURRENCY} (default if not tier/arg)"
echo "--------------------------------------------------------"

# Ensure dependencies are installed
export PYTHONPATH="source:${PYTHONPATH:-}"

CMD=(python3 -m vlm_inference_lab.experiments.benchmark_serving
    --url "${BASE_URL}"
    --model "${MODEL_ID}"
    --num-requests "${NUM_REQUESTS}"
    --concurrency "${CONCURRENCY}"
    --output-dir "${OUTPUT_DIR}")

if [ -n "$TIER" ]; then
    CMD+=(--tier "${TIER}")
fi

CMD+=("${EXTRA_ARGS[@]}")

# Run the benchmark
if ! "${CMD[@]}"; then
    echo "Error: Python benchmark script failed."
    exit 1
fi

echo "--------------------------------------------------------"
echo "Benchmark Complete"
echo "Results saved in: ${OUTPUT_DIR}"
echo "--------------------------------------------------------"
