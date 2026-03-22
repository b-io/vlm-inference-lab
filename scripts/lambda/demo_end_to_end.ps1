# Lambda Labs End-to-End Orchestrator
# This script manages the flow: Start vLLM on Lambda instance -> Benchmark -> Teardown info.

# Support UTF-8 emojis in PowerShell console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 1. Load configuration
if (Test-Path .env.lambda) {
    # Simple .env parsing for PowerShell (assuming KEY=VALUE format)
    Get-Content .env.lambda | Where-Object { $_ -match '=' -and $_ -notmatch '^#' } | ForEach-Object {
        $name, $value = $_.Split('=', 2)
        $name = $name.Trim()
        $value = $value.Trim().Trim('"').Trim("'")
        New-Item -Path "env:\$name" -Value $value -Force | Out-Null
    }
} else {
    Write-Error "Error: .env.lambda not found. Copy .env.lambda.example and fill it in."
    exit 1
}

# Required variables check
if (-not $env:LAMBDA_INSTANCE_IP) { Write-Error "Required LAMBDA_INSTANCE_IP in .env.lambda"; exit 1 }
if (-not $env:LAMBDA_SSH_USER) { Write-Error "Required LAMBDA_SSH_USER in .env.lambda"; exit 1 }
if (-not $env:MODEL_ID) { Write-Error "Required MODEL_ID in .env.lambda"; exit 1 }

$REMOTE_HOST = "$($env:LAMBDA_SSH_USER)@$($env:LAMBDA_INSTANCE_IP)"
$VLLM_PORT = if ($env:VLLM_PORT) { $env:VLLM_PORT } else { 8000 }
$LAMBDA_SSH_PORT = if ($env:LAMBDA_SSH_PORT) { $env:LAMBDA_SSH_PORT } else { 22 }
$HF_TOKEN = if ($env:HF_TOKEN) { $env:HF_TOKEN } else { "" }

Write-Host "--------------------------------------------------------"
Write-Host "VLM Inference Lab: Lambda Labs Orchestrator"
Write-Host "Remote Host : $REMOTE_HOST (Port: $LAMBDA_SSH_PORT)"
Write-Host "Model ID    : ${env:MODEL_ID}"
Write-Host "--------------------------------------------------------"

# 2. Upload startup script to pod
Write-Host "Uploading startup script to instance..."
ssh -o StrictHostKeyChecking=no -p $LAMBDA_SSH_PORT $REMOTE_HOST "mkdir -p ~/vlm-inference-lab"
scp -o StrictHostKeyChecking=no -P $LAMBDA_SSH_PORT scripts/runpod/start_vllm.sh "${REMOTE_HOST}:~/vlm-inference-lab/start_vllm.sh"

# 3. Start vLLM on the pod
Write-Host "Starting vLLM on the instance (this may take time)..."
ssh -o StrictHostKeyChecking=no -p $LAMBDA_SSH_PORT $REMOTE_HOST "bash ~/vlm-inference-lab/start_vllm.sh '${env:MODEL_ID}' '$HF_TOKEN' '$VLLM_PORT'"

# 4. Run the benchmark locally against the remote pod
Write-Host "Running local benchmark against remote instance..."
$BASE_URL = "http://${env:LAMBDA_INSTANCE_IP}:$VLLM_PORT/v1"
.\scripts\benchmark_remote.ps1 "$BASE_URL" "${env:MODEL_ID}"

Write-Host "--------------------------------------------------------"
Write-Host "Demo Successful!"
Write-Host "--------------------------------------------------------"
Write-Host "IMPORTANT: Don't forget to TERMINATE your Lambda instance"
Write-Host "to avoid ongoing charges."
Write-Host "--------------------------------------------------------"
