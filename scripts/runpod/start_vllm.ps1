# Runpod Remote vLLM Startup Script
# Purpose: Start a vLLM server directly or via Docker on a generic GPU VM/pod.

# Support UTF-8 emojis in PowerShell console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# This script is intended for 'generic-remote mode' only.
# Usage: .\start_vllm.ps1 [<model_id>] [<hf_token>] [<port>] [<served_model_name>]

$MODEL_ID = if ($args[0]) { $args[0] } else { throw "Model ID is required" }
$HF_TOKEN = if ($args[1]) { $args[1] } else { "" }
$VLLM_PORT = if ($args[2]) { $args[2] } else { 8000 }
$SERVED_MODEL_NAME = if ($args[3]) { $args[3] } else { "" }

# 1. Prepare startup flags
$VLLM_FLAGS = @("serve", "`"$MODEL_ID`"", "--host", "0.0.0.0", "--port", "$VLLM_PORT", "--trust-remote-code")
if ($SERVED_MODEL_NAME) {
    $VLLM_FLAGS += "--served-model-name", "`"$SERVED_MODEL_NAME`""
}

# 2. Export HF_TOKEN if provided
if ($HF_TOKEN) {
    $env:HF_TOKEN = $HF_TOKEN
}

# 3. Check for local 'vllm' installation
# First, kill any existing vLLM processes to allow redeployment
Write-Host "Checking for existing vLLM processes..."
Get-Process vllm -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
docker stop vllm-server 2>$null | Out-Null
docker rm vllm-server 2>$null | Out-Null
Start-Sleep -Seconds 2

if ($VLLM_CMD = Get-Command vllm -ErrorAction SilentlyContinue) {
    Write-Host "Found local 'vllm' at $($VLLM_CMD.Source). Starting vLLM directly..."
    Write-Host "Command: vllm $($VLLM_FLAGS -join ' ')"
    
    # Start vLLM in the background
    $process = Start-Process vllm -ArgumentList $VLLM_FLAGS `
        -RedirectStandardOutput "vllm_server.log" -RedirectStandardError "vllm_server.err" -PassThru -WindowStyle Hidden
    
    $VLLM_PID = $process.Id
    Write-Host "vLLM started with PID: $VLLM_PID"
    Write-Host "Logs are being written to vllm_server.log and vllm_server.err"

} else {
    # 4. Fallback to Docker
    Write-Host "Local 'vllm' not found. Checking for Docker..."
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Error: Neither 'vllm' nor 'docker' was found on the remote host."
        exit 1
    }

    Write-Host "Cleaning up any existing vllm-server container..."
    docker stop vllm-server 2>$null | Out-Null
    docker rm vllm-server 2>$null | Out-Null

    Write-Host "Starting vLLM via Docker (vllm/vllm-openai:latest)..."
    Write-Host "Model: $MODEL_ID, Port: $VLLM_PORT, Alias: $($SERVED_MODEL_NAME -ifEmpty 'none')"
    
    $home_cache = Join-Path $HOME ".cache\huggingface"
    if (-not (Test-Path $home_cache)) {
        New-Item -ItemType Directory -Force -Path $home_cache | Out-Null
    }
    
    # Prepare Docker args
    $DOCKER_VLLM_FLAGS = $VLLM_FLAGS.Clone()
    for ($i = 0; $i -lt $DOCKER_VLLM_FLAGS.Count; $i++) {
        if ($DOCKER_VLLM_FLAGS[$i] -eq "$VLLM_PORT" -and $DOCKER_VLLM_FLAGS[$i-1] -eq "--port") {
            $DOCKER_VLLM_FLAGS[$i] = "8000"
        }
    }

    docker run -d --name vllm-server `
        --runtime nvidia `
        --gpus all `
        -v "$($home_cache):/root/.cache/huggingface" `
        -p "$($VLLM_PORT):8000" `
        -e HF_TOKEN="$HF_TOKEN" `
        --ipc=host `
        vllm/vllm-openai:latest `
        $DOCKER_VLLM_FLAGS
}

# 5. Health check loop
Write-Host "Waiting for vLLM server to start at http://localhost:$VLLM_PORT/v1/models (this may take time)..."
$MAX_RETRIES = 60
$RETRY_COUNT = 0

$CHECK_MODEL = if ($SERVED_MODEL_NAME) { $SERVED_MODEL_NAME } else { $MODEL_ID }

while ($true) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:$VLLM_PORT/v1/models" -Method Get -TimeoutSec 5 -ErrorAction Stop
        if ($response.data | Where-Object { $_.id -eq $CHECK_MODEL }) {
            Write-Host "vLLM server is UP and READY with model $CHECK_MODEL."
            break
        }
    } catch {
        # Check if the process or container is still running
        if (Get-Command vllm -ErrorAction SilentlyContinue) {
            if (-not (Get-Process -Id $VLLM_PID -ErrorAction SilentlyContinue)) {
                Write-Error "Error: vLLM process died."
                if (Test-Path vllm_server.err) { Get-Content vllm_server.err | Select-Object -Last 20 }
                exit 1
            }
        } else {
            $container = docker ps -q --filter name=vllm-server
            if (-not $container) {
                Write-Error "Error: vllm-server container exited."
                docker logs vllm-server | Select-Object -Last 20
                exit 1
            }
        }
    }

    $RETRY_COUNT++
    if ($RETRY_COUNT -ge $MAX_RETRIES) {
        Write-Error "Error: vLLM server failed to start within $($MAX_RETRIES * 5) seconds."
        if (Get-Command vllm -ErrorAction SilentlyContinue) {
             if (Test-Path vllm_server.log) { Get-Content vllm_server.log | Select-Object -Last 20 }
        } else {
            docker logs vllm-server | Select-Object -Last 20
        }
        exit 1
    }
    
    Write-Host "  - Waiting... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
    Start-Sleep -Seconds 5
}

Write-Host "Endpoint ready at: http://<REMOTE_IP>:$VLLM_PORT/v1"
