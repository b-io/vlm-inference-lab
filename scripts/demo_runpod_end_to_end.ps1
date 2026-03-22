# scripts/demo_runpod_end_to_end.ps1
# Refactored Runpod Orchestrator supporting 'existing' and 'generic' modes.

# Support UTF-8 emojis in PowerShell console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 1. Load Environment Variables from .env.runpod
$EnvFile = ".env.runpod"
if (Test-Path $EnvFile) {
    Write-Host "--- Loading configuration from $EnvFile ---"
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $name, $value = $line.Split('=', 2)
            if ($name -and $value) {
                $value = $value.Trim().Trim('"').Trim("'")
                New-Item -Path "env:\$name" -Value $value -Force | Out-Null
            }
        }
    }
}

# 2. Validate Configuration
$MODE = $env:RUNPOD_MODE
if (-not $MODE) { $MODE = "existing" }

if (-not $env:RUNPOD_BASE_URL) { Write-Error "RUNPOD_BASE_URL is required for both modes"; exit 1 }
if (-not $env:MODEL_ID) { Write-Error "MODEL_ID is required"; exit 1 }

# Helper to validate RUNPOD_BASE_URL
function Assert-ValidRunpodBaseUrl($url, $mode) {
    if ($url -like "*127.0.0.1*" -or $url -like "*localhost*") {
        # Allow localhost for local non-Runpod use
        return
    }
    
    if ($mode -eq "existing") {
        # Reject raw IP + port for existing mode as it usually bypasses the Runpod proxy
        # Pattern matches http:// followed by digits and dots (IP) and then a port
        if ($url -match "^http://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+") {
            Write-Error "In 'existing' mode, RUNPOD_BASE_URL must point to the HTTP service endpoint, typically 'https://<pod-id>-8000.proxy.runpod.net/v1'. Do not use the raw SSH/public IP for HTTP readiness checks."
            exit 1
        }
    }
    
    if ($url -notlike "*.proxy.runpod.net*" -and $url -notlike "*localhost*" -and $url -notlike "*127.0.0.1*") {
        Write-Warning "RUNPOD_BASE_URL ($url) does not appear to be a standard Runpod proxy URL (*.proxy.runpod.net)."
    }
}

Assert-ValidRunpodBaseUrl $env:RUNPOD_BASE_URL $MODE

Write-Host "--------------------------------------------------------"
Write-Host "VLM Inference Lab: Runpod Orchestrator"
Write-Host "Mode        : $MODE"
Write-Host "Model ID    : $($env:MODEL_ID)"
Write-Host "Base URL    : $($env:RUNPOD_BASE_URL)"
Write-Host "--------------------------------------------------------"

if ($MODE -eq "generic") {
    if (-not $env:RUNPOD_SSH_HOST) { Write-Error "RUNPOD_SSH_HOST is required for generic mode"; exit 1 }
    if (-not $env:RUNPOD_SSH_USER) { $env:RUNPOD_SSH_USER = "root" }
    if (-not $env:RUNPOD_SSH_PORT) { $env:RUNPOD_SSH_PORT = "22" }
    if (-not $env:RUNPOD_SSH_KEY_PATH) { $env:RUNPOD_SSH_KEY_PATH = "~/.ssh/id_ed25519" }
}

# 3. Generic Mode: SSH Deployment
if ($MODE -eq "generic") {
    Write-Host "--- Stage: Remote SSH Deployment ---"
    $REMOTE_HOST = "$($env:RUNPOD_SSH_USER)@$($env:RUNPOD_SSH_HOST)"
    $SSH_KEY = $env:RUNPOD_SSH_KEY_PATH
    $SSH_PORT = $env:RUNPOD_SSH_PORT

    Write-Host "Connecting to $REMOTE_HOST (Port: $SSH_PORT)..."

    # Create remote directory
    $SSH_CMD = "mkdir -p ~/vlm-inference-lab/scripts/cloud"
    ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $SSH_PORT $REMOTE_HOST $SSH_CMD
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create remote directory via SSH"; exit 1 }

    # Upload startup script
    Write-Host "Uploading startup script..."
    scp -o StrictHostKeyChecking=no -i $SSH_KEY -P $SSH_PORT scripts/cloud/runpod_start_vllm.sh "${REMOTE_HOST}:~/vlm-inference-lab/scripts/cloud/"
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to SCP startup script"; exit 1 }

    # Start vLLM remotely
    Write-Host "Starting vLLM remotely (this may take time)..."
    $START_CMD = "bash ~/vlm-inference-lab/scripts/cloud/runpod_start_vllm.sh $($env:MODEL_ID) '$($env:HF_TOKEN)' $($env:VLLM_PORT)"
    ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $SSH_PORT $REMOTE_HOST $START_CMD
    if ($LASTEXITCODE -ne 0) { Write-Error "Remote startup script failed"; exit 1 }
}

# 4. Readiness Check (Common for both modes)
Write-Host "--- Stage: Readiness Check ---"

$MAX_RETRIES = 30
$RETRY_COUNT = 0
$HEALTHY = $false
$RESOLVED_MODEL_ID = $env:MODEL_ID
$EXPECTED_MODEL = $env:MODEL_ID
$SHORT_MODEL = $env:MODEL_ID.Split('/')[-1]
$READINESS_URL = "$($env:RUNPOD_BASE_URL)/models"

while ($RETRY_COUNT -lt $MAX_RETRIES) {
    Write-Host "Polling $READINESS_URL ... (Attempt $($RETRY_COUNT + 1)/$MAX_RETRIES)"
    try {
        $Response = Invoke-RestMethod -Uri $READINESS_URL -Method Get -TimeoutSec 5
        if ($Response.data) {
            $Models = $Response.data | ForEach-Object { $_.id }
            if ($Models -contains $EXPECTED_MODEL) {
                Write-Host "vLLM is healthy and model $EXPECTED_MODEL is loaded."
                $RESOLVED_MODEL_ID = $EXPECTED_MODEL
                $HEALTHY = $true
                break
            } elseif ($Models -contains $SHORT_MODEL) {
                Write-Host "vLLM is healthy and model $SHORT_MODEL is loaded (aliased from $EXPECTED_MODEL)."
                $RESOLVED_MODEL_ID = $SHORT_MODEL
                $HEALTHY = $true
                break
            } else {
                Write-Host "Waiting for model $($env:MODEL_ID) to load (Found: $($Models -join ', '))..."
            }
        } else {
             Write-Host "Received response but no model data found."
        }
    } catch {
        $StatusCode = ""
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
            $StatusCode = " [HTTP $($_.Exception.Response.StatusCode.value__)]"
        }
        $Reason = $_.Exception.Message
        Write-Host "Waiting for endpoint to respond...$StatusCode ($Reason)"
    }
    $RETRY_COUNT++
    Start-Sleep -Seconds 10
}

if (-not $HEALTHY) {
    Write-Error "Timeout: vLLM or model failed to become healthy at $READINESS_URL"
    exit 1
}

# 5. Remote Benchmark
Write-Host "--- Stage: Remote Benchmark ---"
Write-Host "Starting benchmark using $env:RUNPOD_BASE_URL (Model: $RESOLVED_MODEL_ID)..."
.\scripts\benchmark_remote.ps1 $env:RUNPOD_BASE_URL $RESOLVED_MODEL_ID 10 2 --fail-on-errors
if ($LASTEXITCODE -ne 0) {
    Write-Error "Benchmark failed"
    exit 1
}

Write-Host "--------------------------------------------------------"
Write-Host "Orchestration Successful!"
Write-Host "--------------------------------------------------------"
Write-Host "IMPORTANT: Don't forget to TERMINATE your Runpod pod"
Write-Host "to avoid ongoing charges for compute and storage."
Write-Host "--------------------------------------------------------"
