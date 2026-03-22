#!/bin/bash
set -euo pipefail

# Run vLLM's internal benchmark tools remotely
# Usage: ./benchmark_vllm_bench.sh <base_url> <model_id> [tier] [extra_args]

BASE_URL=${1:?Required base_url}
MODEL_ID=${2:?Required model_id}
TIER=${3:-smoke}
shift $(( $# >= 3 ? 3 : $# ))

NUM_REQUESTS=10
CONCURRENCY=1

case "$TIER" in
    smoke)
        NUM_REQUESTS=10
        CONCURRENCY=1
        ;;
    latency)
        NUM_REQUESTS=100
        CONCURRENCY=1
        ;;
    throughput)
        NUM_REQUESTS=200
        CONCURRENCY=8
        ;;
esac

OUTPUT_DIR="results/benchmarks/$TIER"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="$OUTPUT_DIR/vllm_bench_$TIMESTAMP.json"

echo "--------------------------------------------------------"
echo "Starting vLLM Professional Benchmark"
echo "URL         : $BASE_URL"
echo "Model       : $MODEL_ID"
echo "Tier        : $TIER"
echo "Requests    : $NUM_REQUESTS"
echo "Concurrency : $CONCURRENCY"
echo "Output      : $OUTPUT_FILE"
echo "--------------------------------------------------------"

# Heuristic for chat models
ENDPOINT="/v1/completions"
BACKEND="openai"
if [[ "$MODEL_ID" == *chat* || "$MODEL_ID" == *instruct* || "$MODEL_ID" == *llama-3* ]]; then
    ENDPOINT="/v1/chat/completions"
    BACKEND="openai-chat"
fi

vllm bench serve \
    --base-url "$BASE_URL" \
    --model "$MODEL_ID" \
    --num-prompts "$NUM_REQUESTS" \
    --max-concurrency "$CONCURRENCY" \
    --endpoint "$ENDPOINT" \
    --backend "$BACKEND" \
    --output-json "$OUTPUT_FILE" \
    "$@"

echo "--------------------------------------------------------"
echo "Benchmark Complete"
echo "--------------------------------------------------------"
