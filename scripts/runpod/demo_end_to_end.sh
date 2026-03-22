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

# New Configuration Variables
RUNPOD_SSH_MODE=${RUNPOD_SSH_MODE:-auto}
RUNPOD_PROXY_SSH_TARGET=${RUNPOD_PROXY_SSH_TARGET:-""}
RUNPOD_ALLOW_MANAGED_INSTANCE_MUTATION=${RUNPOD_ALLOW_MANAGED_INSTANCE_MUTATION:-false}

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

show_ssh_troubleshooting() {
    local message=$1
    local ssh_key=$2
    local strategy=$3
    echo ""
    echo "$message"
    echo ""
    echo "Troubleshooting Runpod SSH (Strategy: $strategy):"
    echo "1. 'generic' mode requires DIRECT TCP SSH access for SCP/SFTP support."
    echo "2. The Runpod 'ssh.runpod.io' shortcut (Proxied SSH) IS NOT SUFFICIENT for SCP/SFTP."
    echo "3. If direct TCP SSH failed, you may need to bootstrap 'sshd' via Proxied SSH."
    echo "4. Use 'RUNPOD_PROXY_SSH_TARGET' in .env.runpod (e.g., j0axhra8c4u1mi-64411b50@ssh.runpod.io)."
    echo "5. Ensure direct TCP host/port are correct from the Connect tab ('SSH over exposed TCP')."
    echo "6. Ensure your SSH key ($ssh_key) is correctly configured."
    echo ""
    echo "If you already have a pod running vLLM, use 'RUNPOD_MODE=existing' instead."
}

is_managed_instance() {
    local base_url=$1
    local remote_host=$2
    local ssh_key=$3
    local ssh_port=$4

    # Check 1: HTTP Readiness
    if curl -s -k -m 5 "${base_url}/models" | grep -q "\"data\":"; then
        return 0
    fi

    # Check 2: Remote Process Check
    if [ -n "$remote_host" ]; then
        if ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i "$ssh_key" -p "$ssh_port" "$remote_host" "pgrep -f 'vllm serve'" >/dev/null 2>&1; then
            return 0
        fi
    fi

    return 1
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
    local found_models
    if command -v jq >/dev/null 2>&1; then
        found_models=$(echo "$body" | jq -r '.data[].id' | tr '\n' ',' | sed 's/,$//')
        if echo "$found_models" | grep -qE "(^|,)$model_id(,|$)"; then
            echo "vLLM is healthy and model ${model_id} is loaded."
            RESOLVED_MODEL_ID="${model_id}"
            return 0
        elif echo "$found_models" | grep -qE "(^|,)$short_model_id(,|$)"; then
            echo "vLLM is healthy and model ${short_model_id} is loaded (aliased from ${model_id})."
            RESOLVED_MODEL_ID="${short_model_id}"
            return 0
        fi
    else
        # Fallback to grep/cut if jq is missing
        if echo "$body" | grep -q "\"id\": \"$model_id\""; then
            echo "vLLM is healthy and model ${model_id} is loaded."
            RESOLVED_MODEL_ID="${model_id}"
            return 0
        elif echo "$body" | grep -q "\"id\": \"$short_model_id\""; then
            echo "vLLM is healthy and model ${short_model_id} is loaded (aliased from ${model_id})."
            RESOLVED_MODEL_ID="${short_model_id}"
            return 0
        fi
        found_models=$(echo "$body" | grep -o '"id": "[^"]*' | cut -d'"' -f4 | tr '\n' ',' | sed 's/,$//')
    fi
    echo "Waiting for model ${model_id} to load (Found: ${found_models:-none})"
    return 1
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
    if ! ./scripts/benchmark_remote.sh "${RUNPOD_BASE_URL}" "${RESOLVED_MODEL_ID}" 10 2 --fail-on-errors; then
        echo "Error: Benchmark failed"
        exit 1
    fi
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
    echo "SSH Strategy: ${RUNPOD_SSH_MODE}"
    echo "--------------------------------------------------------"

    # 1. Detection
    echo "--- Stage: Environment Detection ---"
    if is_managed_instance "${RUNPOD_BASE_URL}" "${remote_host}" "${ssh_key}" "${ssh_port}"; then
        echo "Detected: This host appears to be an already-managed vLLM service pod."
        if [ "${RUNPOD_ALLOW_MANAGED_INSTANCE_MUTATION}" != "true" ]; then
            echo "Error: Generic remote deployment is disabled by default on managed pods to avoid killing the main service process."
            echo "Suggestion: Use 'RUNPOD_MODE=existing', or set 'RUNPOD_ALLOW_MANAGED_INSTANCE_MUTATION=true' only if you are sure."
            exit 1
        fi
    else
        echo "Detected: Generic host (not a managed vLLM service)."
    fi

    # 2. SSH Preflight & Bootstrap
    echo "--- Stage: SSH Preflight & Bootstrap ---"
    local direct_ssh_ok=false
    local scp_ok=false

    # Try Direct SSH
    echo "Direct SSH Preflight Check: Connecting to ${remote_host}:${ssh_port}..."
    if ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i "${ssh_key}" -p "${ssh_port}" "${remote_host}" "echo SSH_OK" > /dev/null 2>&1; then
        direct_ssh_ok=true
        echo "Direct SSH connection verified."
    fi

    # Try Bootstrap if allowed
    if [ "$direct_ssh_ok" = false ] && [[ "${RUNPOD_SSH_MODE}" == "auto" || "${RUNPOD_SSH_MODE}" == "proxied" ]]; then
        if [ -z "${RUNPOD_PROXY_SSH_TARGET}" ]; then
            echo "Direct SSH failed and RUNPOD_PROXY_SSH_TARGET is not configured. Cannot bootstrap."
        else
            echo "Attempting SSH Bootstrap via Proxied SSH (${RUNPOD_PROXY_SSH_TARGET})..."
            local pub_key=""
            if [ -f "${ssh_key}.pub" ]; then pub_key=$(cat "${ssh_key}.pub"); fi
            
            local bootstrap_script
            bootstrap_script=$(cat scripts/runpod/enable_sshd.sh)
            # Use base64 to safely transfer the script over SSH
            local encoded_script
            encoded_script=$(echo "$bootstrap_script" | base64 | tr -d '\n')
            
            if ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no "${RUNPOD_PROXY_SSH_TARGET}" "echo '$encoded_script' | base64 -d > /tmp/bootstrap.sh && bash /tmp/bootstrap.sh '$pub_key'" > /dev/null 2>&1; then
                echo "Bootstrap successful. Retrying direct SSH..."
                if ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i "${ssh_key}" -p "${ssh_port}" "${remote_host}" "echo SSH_OK" > /dev/null 2>&1; then
                    direct_ssh_ok=true
                    echo "Direct SSH now verified."
                fi
            else
                echo "Bootstrap via proxied SSH failed."
            fi
        fi
    fi

    if [ "$direct_ssh_ok" = false ]; then
        show_ssh_troubleshooting "Error: SSH connectivity failed to ${remote_host}:${ssh_port}" "${ssh_key}" "${RUNPOD_SSH_MODE}"
        exit 1
    fi

    # 3. SCP Preflight
    echo "SCP Preflight Check..."
    local temp_file
    temp_file=$(mktemp)
    echo "SCP_TEST" > "${temp_file}"
    local remote_temp="~/scp_test_$RANDOM.tmp"
    
    if scp -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i "${ssh_key}" -P "${ssh_port}" "${temp_file}" "${remote_host}:${remote_temp}" > /dev/null 2>&1; then
        scp_ok=true
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no -i "${ssh_key}" -p "${ssh_port}" "${remote_host}" "rm -f ${remote_temp}" > /dev/null 2>&1
        echo "SCP transfer verified."
    else
        echo "SCP failed. Falling back to SSH-only remote file creation."
    fi
    rm -f "${temp_file}"

    # 4. Deployment
    echo "Creating remote directory..."
    ssh -o StrictHostKeyChecking=no -i "${ssh_key}" -p "${ssh_port}" "${remote_host}" "mkdir -p ~/vlm-inference-lab/scripts/runpod"

    echo "Deploying startup script..."
    if [ "$scp_ok" = true ]; then
        if ! scp -o StrictHostKeyChecking=no -i "${ssh_key}" -P "${ssh_port}" scripts/runpod/start_vllm.sh "${remote_host}:~/vlm-inference-lab/scripts/runpod/"; then
            echo "Error: Failed to SCP startup script to ${remote_host}"
            exit 1
        fi
    else
        # Fallback: Create script via SSH heredoc
        local script_content
        script_content=$(cat scripts/runpod/start_vllm.sh)
        local encoded_script
        encoded_script=$(echo "$script_content" | base64 | tr -d '\n')
        if ! ssh -o StrictHostKeyChecking=no -i "${ssh_key}" -p "${ssh_port}" "${remote_host}" "echo '$encoded_script' | base64 -d > ~/vlm-inference-lab/scripts/runpod/start_vllm.sh && chmod +x ~/vlm-inference-lab/scripts/runpod/start_vllm.sh"; then
             echo "Error: Failed to deploy startup script via SSH fallback"
             exit 1
        fi
    fi

    # 5. Start vLLM
    echo "Starting vLLM on remote host..."
    local short_model="${MODEL_ID##*/}"
    local allow_kill="${RUNPOD_ALLOW_MANAGED_INSTANCE_MUTATION}"
    ssh -o StrictHostKeyChecking=no -i "${ssh_key}" -p "${ssh_port}" "${remote_host}" "bash ~/vlm-inference-lab/scripts/runpod/start_vllm.sh '${MODEL_ID}' '${hf_token}' '${vllm_port}' '${short_model}' '${allow_kill}'"

    # 6. Wait for readiness
    if ! wait_for_vllm_ready "${RUNPOD_BASE_URL}" "${MODEL_ID}"; then
        echo "Error: vLLM did not become healthy at ${RUNPOD_BASE_URL}"
        exit 1
    fi

    # 7. Benchmark
    echo "Starting benchmark using ${RUNPOD_BASE_URL} (Model: ${RESOLVED_MODEL_ID})..."
    if ! ./scripts/benchmark_remote.sh "${RUNPOD_BASE_URL}" "${RESOLVED_MODEL_ID}" 10 2 --fail-on-errors; then
        echo "Error: Benchmark failed"
        exit 1
    fi
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
