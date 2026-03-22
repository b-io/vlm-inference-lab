#!/bin/bash
set -euo pipefail

# Run vLLM parameter sweep remotely
# Usage: ./benchmark_vllm_sweep.sh <base_url> <model_id> [tier] [extra_args]

normalize_runpod_base_url() {
    local url="$1"
    url="${url%/}"
    if [[ "$url" == */v1 ]]; then
        url="${url%/v1}"
    fi
    echo "$url"
}

ORIGINAL_URL=${1:?Required base_url}
BASE_URL=$(normalize_runpod_base_url "$ORIGINAL_URL")
MODEL_ID=${2:?Required model_id}
TIER=${3:-sweep-small}
shift $(( $# >= 3 ? 3 : $# ))

# Remove default sweep parameters as we generate JSON files
OUTPUT_DIR="results/benchmarks/sweeps"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="$OUTPUT_DIR/vllm_sweep_$TIMESTAMP.json"

echo "--------------------------------------------------------"
echo "Starting vLLM Professional Parameter Sweep"
echo "Configured URL : $ORIGINAL_URL"
echo "Normalized URL : $BASE_URL"
echo "Model          : $MODEL_ID"
echo "Output      : $OUTPUT_DIR"
echo "--------------------------------------------------------"

# Check for vLLM sweep compatibility
if vllm bench sweep serve --help 2>&1 | grep -q "--base-url" && \
   vllm bench sweep serve --help 2>&1 | grep -q "--serve-params" && \
   vllm bench sweep serve --help 2>&1 | grep -q "--bench-params"; then
    echo "vLLM sweep CLI compatibility check passed."
else
    echo "Error: Incompatible vLLM version detected for remote sweep."
    echo "The current 'vllm bench sweep serve' command does not appear to support --base-url or JSON parameter files."
    echo "The repo's remote benchmarking path is production-ready for 'bench serve', while remote sweep behavior is version-sensitive."
    echo "Please use a compatible vLLM version (0.4.0+) or stick to 'bench serve' for stable remote benchmarking."
    exit 1
fi

# Generate temporary JSON parameter files
SERVE_PARAMS_FILE="$OUTPUT_DIR/serve_params_$TIMESTAMP.json"
BENCH_PARAMS_FILE="$OUTPUT_DIR/bench_params_$TIMESTAMP.json"

cat <<EOF > "$SERVE_PARAMS_FILE"
{
  "gpu_memory_utilization": [0.85, 0.90, 0.95],
  "max_num_seqs": [16, 32, 64],
  "max_num_batched_tokens": [1024, 2048, 4096]
}
EOF

cat <<EOF > "$BENCH_PARAMS_FILE"
{
  "num_prompts": [100],
  "max_concurrency": [1, 2, 4, 8]
}
EOF

# vllm bench sweep serve --base-url <url> --model <model> --serve-params <file> --bench-params <file>
if vllm bench sweep serve \
    --base-url "$BASE_URL" \
    --model "$MODEL_ID" \
    --serve-params "$SERVE_PARAMS_FILE" \
    --bench-params "$BENCH_PARAMS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    "$@"; then
    
    echo "--------------------------------------------------------"
    echo "Sweep Complete. Generating Pareto Summary..."
    export PYTHONPATH="source:${PYTHONPATH:-}"
    python3 -m vlm_inference_lab.experiments.pareto_analysis --input-dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/pareto_summary_$TIMESTAMP.md"
else
    echo "Error: vLLM sweep failed."
    exit 1
fi
