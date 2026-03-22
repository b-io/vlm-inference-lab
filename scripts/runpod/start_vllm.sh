#!/bin/bash
set -euo pipefail

# Runpod Remote vLLM Startup Script
# Purpose: Start a vLLM server directly or via Docker on a generic GPU VM/pod.
# This script is intended for 'generic-remote mode' only.
# Usage: ./start_vllm.sh <model_id> <hf_token> [port] [served_model_name] [allow_kill]

MODEL_ID=${1:?Model ID is required}
HF_TOKEN=${2:-""}
VLLM_PORT=${3:-8000}
SERVED_MODEL_NAME=${4:-""}
ALLOW_KILL=${5:-false}

# 1. Managed-instance safety guard
if [ "$ALLOW_KILL" != "true" ]; then
    if pgrep -f "vllm serve" >/dev/null 2>&1; then
        echo "-----------------------------------------------------------------------"
        echo "Error: 'vllm serve' is already running and ALLOW_KILL is false."
        echo "This target appears to already be serving vLLM."
        echo "To avoid killing the container’s main service, generic mutation is blocked by default."
        echo "Recommendation: Use 'RUNPOD_MODE=existing' for benchmark-only operation,"
        echo "or explicitly allow mutation only on a generic SSH-managed host by setting"
        echo "RUNPOD_ALLOW_KILL_EXISTING_VLLM=true in your .env.runpod file."
        echo "-----------------------------------------------------------------------"
        exit 1
    fi
fi

# 2. Cleanup (only if allowed or safe)
echo "Checking for existing vLLM processes..."
CURRENT_PID=$$
# We look for the exact 'vllm serve' executable pattern, but exclude the current process PID
PIDS_TO_KILL=$(pgrep -f "vllm serve" | grep -v "^${CURRENT_PID}$" || true)

if [ -n "$PIDS_TO_KILL" ]; then
    if [ "$ALLOW_KILL" = "true" ]; then
        echo "Killing existing vLLM PIDs: $PIDS_TO_KILL"
        echo "$PIDS_TO_KILL" | xargs kill -9 || true
    else
        echo "Warning: Existing vLLM processes found ($PIDS_TO_KILL) but ALLOW_KILL is false. Skipping kill."
    fi
fi

if [ "$ALLOW_KILL" = "true" ]; then
    docker stop vllm-server &> /dev/null || true
    docker rm vllm-server &> /dev/null || true
fi
sleep 2

# 3. Prepare startup flags
VLLM_FLAGS=(
    "${MODEL_ID}"
    --host 0.0.0.0
    --port "${VLLM_PORT}"
    --trust-remote-code
)

if [ -n "${SERVED_MODEL_NAME}" ]; then
    VLLM_FLAGS+=(--served-model-name "${SERVED_MODEL_NAME}")
fi

# 2. Export HF_TOKEN if provided
if [ -n "${HF_TOKEN}" ]; then
    export HF_TOKEN="${HF_TOKEN}"
fi

# 4. Check for local 'vllm' installation
if command -v vllm &> /dev/null; then
    VLLM_PATH=$(command -v vllm)
    echo "Found local 'vllm' at ${VLLM_PATH}. Starting vLLM directly..."
    echo "Command: vllm serve ${VLLM_FLAGS[*]}"
    
    # Start vLLM in the background
    nohup vllm serve "${VLLM_FLAGS[@]}" > vllm_server.log 2>&1 &
    
    VLLM_PID=$!
    echo "vLLM started with PID: ${VLLM_PID}"
    echo "Logs are being written to vllm_server.log"

else
    # 5. Fallback to Docker
    echo "Local 'vllm' not found. Checking for Docker..."
    if ! command -v docker &> /dev/null; then
        echo "Error: Neither 'vllm' nor 'docker' was found on the remote host."
        exit 1
    fi

    if [ "$ALLOW_KILL" = "true" ]; then
        echo "Cleaning up any existing vllm-server container..."
        docker stop vllm-server &> /dev/null || true
        docker rm vllm-server &> /dev/null || true
    fi

    echo "Starting vLLM via Docker (vllm/vllm-openai:latest)..."
    echo "Model: ${MODEL_ID}, Port: ${VLLM_PORT}, Alias: ${SERVED_MODEL_NAME:-none}"
    mkdir -p ~/.cache/huggingface
    
    # Docker internal port is always 8000 in this command, mapped to host VLLM_PORT
    DOCKER_VLLM_FLAGS=("${VLLM_FLAGS[@]}")
    # Replace the host port with 8000 for the internal container process
    for i in "${!DOCKER_VLLM_FLAGS[@]}"; do
        if [ "${DOCKER_VLLM_FLAGS[$i]}" == "${VLLM_PORT}" ] && [ "${DOCKER_VLLM_FLAGS[$((i-1))]}" == "--port" ]; then
            DOCKER_VLLM_FLAGS[$i]=8000
        fi
    done

    docker run -d --name vllm-server \
        --runtime nvidia \
        --gpus all \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -p "${VLLM_PORT}:8000" \
        -e HF_TOKEN="${HF_TOKEN}" \
        --ipc=host \
        vllm/vllm-openai:latest \
        serve "${DOCKER_VLLM_FLAGS[@]}"
fi

# 6. Health check loop
# We check on localhost:VLLM_PORT if direct, or localhost:VLLM_PORT (mapped) if Docker.
# Note: In both cases, the external-facing port is VLLM_PORT.
echo "Waiting for vLLM server to start at http://localhost:${VLLM_PORT}/v1/models (this may take time)..."
MAX_RETRIES=60
RETRY_COUNT=0

CHECK_MODEL="${SERVED_MODEL_NAME:-${MODEL_ID}}"

while true; do
    # Try to fetch models and check for the specific model ID or alias
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" | grep -q "${CHECK_MODEL}"; then
        echo "vLLM server is UP and READY with model ${CHECK_MODEL}."
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
