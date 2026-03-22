#!/bin/bash
set -euo pipefail

# Runpod Remote vLLM Startup Script
# Purpose: Start a vLLM server directly or via Docker on a generic GPU VM/pod.
# This script is intended for 'generic-remote mode' only.
# Usage: ./runpod_start_vllm.sh <model_id> <hf_token> [port]

MODEL_ID=${1:?Model ID is required}
HF_TOKEN=${2:-""}
VLLM_PORT=${3:-8000}

# 1. Export HF_TOKEN if provided
if [ -n "${HF_TOKEN}" ]; then
    export HF_TOKEN="${HF_TOKEN}"
fi

# 2. Check for local 'vllm' installation
# First, kill any existing vLLM processes to allow redeployment
echo "Checking for existing vLLM processes..."
pkill -f "vllm serve" || true
docker stop vllm-server &> /dev/null || true
docker rm vllm-server &> /dev/null || true
sleep 2

if command -v vllm &> /dev/null; then
    echo "Found local 'vllm' installation. Starting vLLM directly..."
    
    # Start vLLM in the background
    nohup vllm serve "${MODEL_ID}" --host 0.0.0.0 --port "${VLLM_PORT}" --trust-remote-code > vllm_server.log 2>&1 &
    
    VLLM_PID=$!
    echo "vLLM started with PID: ${VLLM_PID}"
    echo "Logs are being written to vllm_server.log"

else
    # 3. Fallback to Docker
    echo "Local 'vllm' not found. Checking for Docker..."
    if ! command -v docker &> /dev/null; then
        echo "Error: Neither 'vllm' nor 'docker' was found on the remote host."
        exit 1
    fi

    echo "Cleaning up any existing vllm-server container..."
    docker stop vllm-server &> /dev/null || true
    docker rm vllm-server &> /dev/null || true

    echo "Starting vLLM via Docker (vllm/vllm-openai:latest)..."
    mkdir -p ~/.cache/huggingface
    
    docker run -d --name vllm-server \
        --runtime nvidia \
        --gpus all \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -p "${VLLM_PORT}:8000" \
        -e HF_TOKEN="${HF_TOKEN}" \
        --ipc=host \
        vllm/vllm-openai:latest \
        serve "${MODEL_ID}" \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code
fi

# 4. Health check loop
# We check on localhost:VLLM_PORT if direct, or localhost:VLLM_PORT (mapped) if Docker.
# Note: In both cases, the external-facing port is VLLM_PORT.
echo "Waiting for vLLM server to start at http://localhost:${VLLM_PORT}/v1/models (this may take time)..."
MAX_RETRIES=60
RETRY_COUNT=0

while true; do
    # Try to fetch models and check for the specific model ID
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" | grep -q "${MODEL_ID}"; then
        echo "vLLM server is UP and READY with model ${MODEL_ID}."
        break
    fi
    
    # Check if the process or container is still running
    if command -v vllm &> /dev/null; then
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "Error: vLLM process died. Check vllm_server.log"
            tail -n 20 vllm_server.log
            exit 1
        fi
    else
        if ! docker ps -q --filter name=vllm-server | grep -q .; then
            echo "Error: vllm-server container exited."
            docker logs vllm-server | tail -n 20
            exit 1
        fi
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Error: vLLM server failed to start within $((MAX_RETRIES * 5)) seconds."
        if command -v vllm &> /dev/null; then
            tail -n 20 vllm_server.log
        else
            docker logs vllm-server | tail -n 20
        fi
        exit 1
    fi
    
    echo "  - Waiting... (Attempt ${RETRY_COUNT}/${MAX_RETRIES})"
    sleep 5
done

echo "Endpoint ready at: http://<REMOTE_IP>:${VLLM_PORT}/v1"
