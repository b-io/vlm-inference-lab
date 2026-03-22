# Runpod Remote vLLM Startup Script
# Purpose: Start a vLLM server directly or via Docker on a generic GPU VM/pod.

# Support UTF-8 emojis in PowerShell console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# This script is intended for 'generic-remote mode' only.
# Usage: .\runpod_start_vllm.ps1 [<model_id>] [<hf_token>] [<port>]

$MODEL_ID = if ($args[0]) { $args[0] } else { throw "Model ID is required" }
$HF_TOKEN = if ($args[1]) { $args[1] } else { "" }
$VLLM_PORT = if ($args[2]) { $args[2] } else { 8000 }

# 1. Export HF_TOKEN if provided
if ($HF_TOKEN) {
    $env:HF_TOKEN = $HF_TOKEN
}

# 2. Check for local 'vllm' installation
if (Get-Command vllm -ErrorAction SilentlyContinue) {
    Write-Host "Found local 'vllm' installation. Starting vLLM directly..."
    
    # Start vLLM in the background
    # On Windows/PowerShell, we can use Start-Process
    $process = Start-Process vllm -ArgumentList "serve", "`"$MODEL_ID`"", "--host", "0.0.0.0", "--port", "$VLLM_PORT", "--trust-remote-code" `
        -RedirectStandardOutput "vllm_server.log" -RedirectStandardError "vllm_server.err" -PassThru -WindowStyle Hidden
    
    $VLLM_PID = $process.Id
    Write-Host "vLLM started with PID: $VLLM_PID"
    Write-Host "Logs are being written to vllm_server.log and vllm_server.err"

} else {
    # 3. Fallback to Docker
    Write-Host "Local 'vllm' not found. Checking for Docker..."
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Error: Neither 'vllm' nor 'docker' was found on the remote host."
        exit 1
    }

    Write-Host "Cleaning up any existing vllm-server container..."
    docker stop vllm-server 2>$null | Out-Null
    docker rm vllm-server 2>$null | Out-Null

    Write-Host "Starting vLLM via Docker (vllm/vllm-openai:latest)..."
    $home_cache = Join-Path $HOME ".cache\huggingface"
    if (-not (Test-Path $home_cache)) {
        New-Item -ItemType Directory -Force -Path $home_cache | Out-Null
    }
    
    docker run -d --name vllm-server `
        --runtime nvidia `
        --gpus all `
        -v "$($home_cache):/root/.cache/huggingface" `
        -p "$($VLLM_PORT):8000" `
        -e HF_TOKEN="$HF_TOKEN" `
        --ipc=host `
        vllm/vllm-openai:latest `
        serve "$MODEL_ID" `
        --host 0.0.0.0 `
        --port 8000 `
        --trust-remote-code
}

# 4. Health check loop
Write-Host "Waiting for vLLM server to start at http://localhost:$VLLM_PORT/v1/models (this may take time)..."
$MAX_RETRIES = 60
$RETRY_COUNT = 0

while ($true) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:$VLLM_PORT/v1/models" -Method Get -TimeoutSec 5 -ErrorAction Stop
        if ($response.data | Where-Object { $_.id -eq $MODEL_ID }) {
            Write-Host "vLLM server is UP and READY with model $MODEL_ID."
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
