#!/bin/bash
set -euo pipefail

# Runpod End-to-End Orchestrator
# This script manages the flow: Start vLLM on pod (if generic) -> Benchmark -> Teardown info.

# 1. Load configuration
if [ -f .env.runpod ]; then
    set -a
    source .env.runpod
    set +a
else
    echo "Error: .env.runpod not found. Copy .env.runpod.example and fill it in."
    exit 1
fi

# 2. Validate Configuration
RUNPOD_MODE=${RUNPOD_MODE:-existing}
MODEL_ID=${MODEL_ID:?Required MODEL_ID in .env.runpod}
RUNPOD_BASE_URL=${RUNPOD_BASE_URL:?Required RUNPOD_BASE_URL in .env.runpod}

validate_base_url() {
    local url=$1
    local mode=$2

    if [[ "$url" == *"127.0.0.1"* ]] || [[ "$url" == *"localhost"* ]]; then
        return 0
    fi

    if [ "$mode" = "existing" ]; then
        # Reject raw IP + port for existing mode
        if [[ "$url" =~ ^http://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+ ]]; then
            echo "Error: In 'existing' mode, RUNPOD_BASE_URL must point to the HTTP service endpoint, typically 'https://<pod-id>-8000.proxy.runpod.net/v1'. Do not use the raw SSH/public IP for HTTP readiness checks."
            exit 1
        fi
    fi

    if [[ "$url" != *".proxy.runpod.net"* ]] && [[ "$url" != *"localhost"* ]] && [[ "$url" != *"127.0.0.1"* ]]; then
        echo "Warning: RUNPOD_BASE_URL ($url) does not appear to be a standard Runpod proxy URL (*.proxy.runpod.net)."
    fi
}

validate_base_url "${RUNPOD_BASE_URL}" "${RUNPOD_MODE}"

test_vllm_health() {
    local base_url=$1
    local model_id=$2
    local readines_url="${base_url}/models"
    echo "Polling ${readines_url} ..."

    # Try to fetch models JSON
    local response
    local http_code
    response=$(curl -s -k -w "\n%{http_code}" -m 10 "${readines_url}") || {
        echo "Backend is not responding yet (Network error or timeout)."
        return 1
    }

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)

    if [ "$http_code" -ne 200 ]; then
        echo "Received HTTP ${http_code} from ${readines_url}."
        return 1
    fi

    # Verify expected model is present
    local short_model_id="${model_id##*/}"
    if echo "$body" | grep -q "\"id\": \"$model_id\""; then
        echo "vLLM is healthy and model ${model_id} is loaded."
        RESOLVED_MODEL_ID="${model_id}"
        return 0
    elif echo "$body" | grep -q "\"id\": \"$short_model_id\""; then
        echo "vLLM is healthy and model ${short_model_id} is loaded (aliased from ${model_id})."
        RESOLVED_MODEL_ID="${short_model_id}"
        return 0
    else
        local found_models
        found_models=$(echo "$body" | grep -o '"id": "[^"]*' | cut -d'"' -f4 | tr '\n' ',' | sed 's/,$//')
        echo "Waiting for model ${model_id} to load (Found: ${found_models:-none})"
        return 1
    fi
}

wait_for_vllm_ready() {
    local base_url=$1
    local model_id=$2
    local max_retries=30
    local delay=10
    RESOLVED_MODEL_ID="${model_id}"
    echo "Waiting for vLLM to be ready at ${base_url}..."
    for ((i=1; i<=max_retries; i++)); do
        if test_vllm_health "${base_url}" "${model_id}"; then
            return 0
        fi
        echo "Retry $i/$max_retries. Waiting $delay seconds..."
        sleep $delay
    done
    return 1
}

invoke_existing_mode() {
    echo "Mode: Existing Service"
    echo "Base URL : ${RUNPOD_BASE_URL}"
    echo "Model ID : ${MODEL_ID}"
    echo "--------------------------------------------------------"

    if ! wait_for_vllm_ready "${RUNPOD_BASE_URL}" "${MODEL_ID}"; then
        echo "Error: vLLM did not become healthy at ${RUNPOD_BASE_URL}"
        exit 1
    fi

    echo "Starting benchmark using ${RUNPOD_BASE_URL} (Model: ${RESOLVED_MODEL_ID})..."
    ./scripts/benchmark_remote.sh "${RUNPOD_BASE_URL}" "${RESOLVED_MODEL_ID}" --fail-on-errors
}

invoke_generic_mode() {
    local ssh_host=${RUNPOD_SSH_HOST:?Required RUNPOD_SSH_HOST in .env.runpod for generic mode}
    local ssh_user=${RUNPOD_SSH_USER:-root}
    local ssh_port=${RUNPOD_SSH_PORT:-22}
    local ssh_key=${RUNPOD_SSH_KEY_PATH:-~/.ssh/id_ed25519}
    local vllm_port=${VLLM_PORT:-8000}
    local hf_token=${HF_TOKEN:-""}

    local remote_host="${ssh_user}@${ssh_host}"

    echo "Mode: Generic Remote (SSH)"
    echo "Remote Host : ${remote_host} (Port: ${ssh_port})"
    echo "Base URL    : ${RUNPOD_BASE_URL}"
    echo "Model ID    : ${MODEL_ID}"
    echo "--------------------------------------------------------"

    # 1. Create remote directory
    echo "Creating remote directory..."
    ssh -o StrictHostKeyChecking=no -i "${ssh_key}" -p "${ssh_port}" "${remote_host}" "mkdir -p ~/vlm-inference-lab/scripts/cloud"

    # 2. Upload startup script
    echo "Uploading startup script..."
    scp -o StrictHostKeyChecking=no -i "${ssh_key}" -P "${ssh_port}" scripts/cloud/runpod_start_vllm.sh "${remote_host}:~/vlm-inference-lab/scripts/cloud/"

    # 3. Start vLLM
    echo "Starting vLLM on remote host..."
    ssh -o StrictHostKeyChecking=no -i "${ssh_key}" -p "${ssh_port}" "${remote_host}" "bash ~/vlm-inference-lab/scripts/cloud/runpod_start_vllm.sh '${MODEL_ID}' '${hf_token}' '${vllm_port}'"

    # 4. Wait for readiness
    if ! wait_for_vllm_ready "${RUNPOD_BASE_URL}" "${MODEL_ID}"; then
        echo "Error: vLLM did not become healthy at ${RUNPOD_BASE_URL}"
        exit 1
    fi

    # 5. Benchmark
    echo "Starting benchmark using ${RUNPOD_BASE_URL} (Model: ${RESOLVED_MODEL_ID})..."
    ./scripts/benchmark_remote.sh "${RUNPOD_BASE_URL}" "${RESOLVED_MODEL_ID}" --fail-on-errors
}

# --- Main Execution ---

echo "--------------------------------------------------------"
echo "VLM Inference Lab: Runpod Orchestrator"

if [ "${RUNPOD_MODE}" = "existing" ]; then
    invoke_existing_mode
elif [ "${RUNPOD_MODE}" = "generic" ]; then
    invoke_generic_mode
else
    echo "Error: Invalid RUNPOD_MODE: ${RUNPOD_MODE}. Must be 'existing' or 'generic'."
    exit 1
fi

echo "--------------------------------------------------------"
echo "Demo Successful!"
echo "--------------------------------------------------------"
echo "IMPORTANT: Don't forget to TERMINATE your Runpod pod"
echo "to avoid ongoing charges for compute and storage."
echo "--------------------------------------------------------"
