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
}

show_ssh_troubleshooting() {
    local message=$1
    local ssh_key=$2
    local strategy=$3
    echo ""
    echo "$message"
    echo ""
    echo "Troubleshooting Runpod SSH (Strategy: $strategy):"
    echo "1. 'generic' mode requires shell access for deployment."
    echo "2. Proxied SSH (ssh.runpod.io) is supported even if direct TCP SSH is blocked."
    echo "3. Ensure 'RUNPOD_PROXY_SSH_TARGET' is set in .env.runpod (e.g. user@ssh.runpod.io)."
    echo "4. If direct TCP failed, ensure your SSH key ($ssh_key) is correctly configured."
    echo "5. If you already have a pod running vLLM, use 'RUNPOD_MODE=existing' instead."
    echo ""
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

    # Check 2: Remote Process Check (if direct SSH works)
    if [ -n "$remote_host" ] && [ -f "$ssh_key" ]; then
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

    local tier=${BENCHMARK_TIER:-smoke}
    local benchmark_path=${BENCHMARK_PATH:-default}

    if [ "$benchmark_path" = "vllm" ]; then
        echo "Starting professional benchmark (vllm bench) using ${RUNPOD_BASE_URL} (Model: ${RESOLVED_MODEL_ID})..."
        if ! ./scripts/benchmark_vllm_bench.sh "${RUNPOD_BASE_URL}" "${RESOLVED_MODEL_ID}" "$tier"; then
            echo "Error: Benchmark failed"
            exit 1
        fi
    elif [ "$benchmark_path" = "sweep" ]; then
        echo "Starting professional sweep (vllm bench sweep) using ${RUNPOD_BASE_URL} (Model: ${RESOLVED_MODEL_ID})..."
        if ! ./scripts/benchmark_vllm_sweep.sh "${RUNPOD_BASE_URL}" "${RESOLVED_MODEL_ID}" "$tier"; then
            echo "Error: Sweep failed"
            exit 1
        fi
    else
        echo "Starting benchmark ($tier) using ${RUNPOD_BASE_URL} (Model: ${RESOLVED_MODEL_ID})..."
        if ! ./scripts/benchmark_remote.sh "${RUNPOD_BASE_URL}" "${RESOLVED_MODEL_ID}" --tier "$tier" --fail-on-errors; then
            echo "Error: Benchmark failed"
            exit 1
        fi
    fi
}

invoke_generic_mode() {
    local ssh_host=${RUNPOD_SSH_HOST:-""}
    local ssh_user=${RUNPOD_SSH_USER:-root}
    local ssh_port=${RUNPOD_SSH_PORT:-22}
    local ssh_key=${RUNPOD_SSH_KEY_PATH:-~/.ssh/id_ed25519}
    local vllm_port=${VLLM_PORT:-8000}
    local hf_token=${HF_TOKEN:-""}

    local remote_host="${ssh_user}@${ssh_host}"

    if [ -z "$ssh_host" ] && [ "$RUNPOD_SSH_MODE" != "proxied" ]; then
        echo "Error: Required RUNPOD_SSH_HOST in .env.runpod for generic mode (unless RUNPOD_SSH_MODE=proxied)"
        exit 1
    fi

    echo "Mode: Generic Remote (SSH)"
    echo "Base URL    : ${RUNPOD_BASE_URL}"
    echo "Model ID    : ${MODEL_ID}"
    echo "SSH Strategy: ${RUNPOD_SSH_MODE}"
    echo "--------------------------------------------------------"

    # 1. Detection
    echo "--- Stage: Environment Detection ---"
    if is_managed_instance "${RUNPOD_BASE_URL}" "${remote_host}" "${ssh_key}" "${ssh_port}"; then
        echo "Detected: This target appears to already be serving vLLM."
        if [ "${RUNPOD_ALLOW_MANAGED_INSTANCE_MUTATION}" != "true" ]; then
            echo "-----------------------------------------------------------------------"
            echo "Error: Generic mutation is blocked by default on managed pods to avoid"
            echo "killing the container’s main service process."
            echo ""
            echo "Recommendation: Use 'RUNPOD_MODE=existing' for benchmark-only operation,"
            echo "or set 'RUNPOD_ALLOW_MANAGED_INSTANCE_MUTATION=true' only if you are sure."
            echo "-----------------------------------------------------------------------"
            exit 1
        fi
    else
        echo "Detected: Generic host (not a managed vLLM service)."
    fi

    # 2. SSH Transport Decision
    echo "--- Stage: SSH Deployment ---"
    local deploy_target=""
    local deploy_opts=("-o" "BatchMode=yes" "-o" "ConnectTimeout=10" "-o" "StrictHostKeyChecking=no")
    local scp_ok=false
    local direct_ssh="unavailable"
    local proxied_ssh="unavailable"
    local scp_support="unavailable"
    local fallback_mode="none"

    if [ "${RUNPOD_SSH_MODE}" = "proxied" ]; then
        if [ -z "${RUNPOD_PROXY_SSH_TARGET}" ]; then
            echo "Error: RUNPOD_PROXY_SSH_TARGET is required for proxied mode."
            exit 1
        fi
        deploy_target="${RUNPOD_PROXY_SSH_TARGET}"
        proxied_ssh="available"
        echo "Using Proxied SSH transport: ${deploy_target}"
    else
        # Try Direct first
        echo "Attempting Direct SSH Preflight (${remote_host})..."
        if ssh "${deploy_opts[@]}" -i "${ssh_key}" -p "${ssh_port}" "${remote_host}" "echo SSH_OK" > /dev/null 2>&1; then
            deploy_target="${remote_host}"
            deploy_opts+=("-i" "${ssh_key}" "-p" "${ssh_port}")
            direct_ssh="available"
            echo "Direct SSH verified."
            
            # Try SCP preflight
            local temp_file
            temp_file=$(mktemp)
            echo "SCP_TEST" > "${temp_file}"
            if scp "${deploy_opts[@]}" -P "${ssh_port}" "${temp_file}" "${remote_host}:/tmp/scp_test.tmp" > /dev/null 2>&1; then
                scp_ok=true
                scp_support="available"
                ssh "${deploy_opts[@]}" "${remote_host}" "rm /tmp/scp_test.tmp"
                echo "SCP transfer verified."
            else
                fallback_mode="SSH-only file creation"
                echo "SCP failed. Will use SSH-only fallback."
            fi
            rm -f "${temp_file}"
        elif [[ "${RUNPOD_SSH_MODE}" == "auto" && -n "${RUNPOD_PROXY_SSH_TARGET}" ]]; then
            echo "Direct SSH failed. Attempting Proxied SSH fallback (${RUNPOD_PROXY_SSH_TARGET})..."
            if ssh "${deploy_opts[@]}" "${RUNPOD_PROXY_SSH_TARGET}" "echo SSH_OK" > /dev/null 2>&1; then
                deploy_target="${RUNPOD_PROXY_SSH_TARGET}"
                proxied_ssh="available"
                fallback_mode="SSH-only file creation"
                echo "Proxied SSH verified."
            fi
        fi
    fi

    echo ""
    echo "Connection Summary:"
    echo "  Direct SSH: ${direct_ssh}"
    echo "  Proxied SSH: ${proxied_ssh}"
    echo "  SCP Support: ${scp_support}"
    echo "  Fallback mode (SSH-only): $([ "$fallback_mode" != "none" ] && echo "yes" || echo "no")"
    echo ""

    if [ -z "$deploy_target" ]; then
        show_ssh_troubleshooting "Error: SSH connectivity failed." "${ssh_key}" "${RUNPOD_SSH_MODE}"
        exit 1
    fi

    # 3. Deployment
    echo "Creating remote directory..."
    ssh "${deploy_opts[@]}" "${deploy_target}" "mkdir -p ~/vlm-inference-lab/scripts/runpod"

    echo "Deploying startup script..."
    if [ "$scp_ok" = true ]; then
        scp "${deploy_opts[@]}" -P "${ssh_port}" scripts/runpod/start_vllm.sh "${deploy_target}:~/vlm-inference-lab/scripts/runpod/start_vllm.sh"
    else
        # Fallback: Create script via SSH heredoc using base64 for safety
        local script_content
        script_content=$(cat scripts/runpod/start_vllm.sh)
        local encoded_script
        encoded_script=$(echo "$script_content" | base64 | tr -d '\n')
        ssh "${deploy_opts[@]}" "${deploy_target}" "echo '$encoded_script' | base64 -d > ~/vlm-inference-lab/scripts/runpod/start_vllm.sh && chmod +x ~/vlm-inference-lab/scripts/runpod/start_vllm.sh"
    fi

    # 4. Start vLLM
    echo "Starting vLLM on remote host..."
    local allow_kill=${RUNPOD_ALLOW_KILL_EXISTING_VLLM:-false}
    ssh "${deploy_opts[@]}" "${deploy_target}" "bash ~/vlm-inference-lab/scripts/runpod/start_vllm.sh '${MODEL_ID}' '${hf_token}' '${vllm_port}' '' '${allow_kill}'"

    # 5. Wait for readiness
    if ! wait_for_vllm_ready "${RUNPOD_BASE_URL}" "${MODEL_ID}"; then
        echo "Error: vLLM did not become healthy at ${RUNPOD_BASE_URL}"
        exit 1
    fi

    local tier=${BENCHMARK_TIER:-smoke}
    local benchmark_path=${BENCHMARK_PATH:-default}

    if [ "$benchmark_path" = "vllm" ]; then
        echo "Starting professional benchmark (vllm bench) using ${RUNPOD_BASE_URL} (Model: ${RESOLVED_MODEL_ID})..."
        if ! ./scripts/benchmark_vllm_bench.sh "${RUNPOD_BASE_URL}" "${RESOLVED_MODEL_ID}" "$tier"; then
            echo "Error: Benchmark failed"
            exit 1
        fi
    elif [ "$benchmark_path" = "sweep" ]; then
        echo "Starting professional sweep (vllm bench sweep) using ${RUNPOD_BASE_URL} (Model: ${RESOLVED_MODEL_ID})..."
        if ! ./scripts/benchmark_vllm_sweep.sh "${RUNPOD_BASE_URL}" "${RESOLVED_MODEL_ID}" "$tier"; then
            echo "Error: Sweep failed"
            exit 1
        fi
    else
        echo "Starting benchmark ($tier) using ${RUNPOD_BASE_URL} (Model: ${RESOLVED_MODEL_ID})..."
        if ! ./scripts/benchmark_remote.sh "${RUNPOD_BASE_URL}" "${RESOLVED_MODEL_ID}" --tier "$tier" --fail-on-errors; then
            echo "Error: Benchmark failed"
            exit 1
        fi
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
echo "Tier        : ${BENCHMARK_TIER:-smoke}"
echo "IMPORTANT: Don't forget to TERMINATE your Runpod pod"
echo "to avoid ongoing charges for compute and storage."
echo "--------------------------------------------------------"
