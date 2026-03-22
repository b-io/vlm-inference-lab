#!/bin/bash
set -euo pipefail

# Lambda Labs End-to-End Orchestrator
# This script manages the flow: Start vLLM on Lambda instance -> Benchmark -> Teardown info.

# 1. Load configuration
if [ -f .env.lambda ]; then
    source .env.lambda
else
    echo "Error: .env.lambda not found. Copy .env.lambda.example and fill it in."
    exit 1
fi

# Required variables check
: "${LAMBDA_INSTANCE_IP:?Required LAMBDA_INSTANCE_IP in .env.lambda}"
: "${LAMBDA_SSH_USER:?Required LAMBDA_SSH_USER in .env.lambda}"
: "${MODEL_ID:?Required MODEL_ID in .env.lambda}"

REMOTE_HOST="${LAMBDA_SSH_USER}@${LAMBDA_INSTANCE_IP}"
VLLM_PORT=${VLLM_PORT:-8000}
LAMBDA_SSH_PORT=${LAMBDA_SSH_PORT:-22}
HF_TOKEN=${HF_TOKEN:-""}

echo "--------------------------------------------------------"
echo "VLM Inference Lab: Lambda Labs Orchestrator"
echo "Remote Host : ${REMOTE_HOST} (Port: ${LAMBDA_SSH_PORT})"
echo "Model ID    : ${MODEL_ID}"
echo "--------------------------------------------------------"

# 2. Upload startup script to pod
echo "Uploading startup script to instance..."
ssh -o StrictHostKeyChecking=no -p "${LAMBDA_SSH_PORT}" "${REMOTE_HOST}" "mkdir -p ~/vlm-inference-lab"
scp -o StrictHostKeyChecking=no -P "${LAMBDA_SSH_PORT}" scripts/cloud/runpod_start_vllm.sh "${REMOTE_HOST}:~/vlm-inference-lab/"

# 3. Start vLLM on the pod
echo "Starting vLLM on the instance (this may take time)..."
ssh -o StrictHostKeyChecking=no -p "${LAMBDA_SSH_PORT}" "${REMOTE_HOST}" "bash ~/vlm-inference-lab/runpod_start_vllm.sh '${MODEL_ID}' '${HF_TOKEN}' '${VLLM_PORT}'"

# 4. Run the benchmark locally against the remote pod
echo "Running local benchmark against remote instance..."
BASE_URL="http://${LAMBDA_INSTANCE_IP}:${VLLM_PORT}/v1"
./scripts/benchmark_remote.sh "${BASE_URL}" "${MODEL_ID}"

echo "--------------------------------------------------------"
echo "Demo Successful!"
echo "--------------------------------------------------------"
echo "IMPORTANT: Don't forget to TERMINATE your Lambda instance"
echo "to avoid ongoing charges."
echo "--------------------------------------------------------"
