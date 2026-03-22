# Runpod Diagnostics Script
# Purpose: Check connectivity, SSH, and model serving status

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "--- Runpod Diagnostics ---" -ForegroundColor Cyan

# 1. Environment Variables
Write-Host "[1/4] Environment Checks" -ForegroundColor Yellow
$MODE = $env:RUNPOD_MODE
if (-not $MODE) { $MODE = "existing" }
Write-Host "  RUNPOD_MODE     : $MODE"
Write-Host "  RUNPOD_BASE_URL : $($env:RUNPOD_BASE_URL)"
Write-Host "  MODEL_ID        : $($env:MODEL_ID)"

# 2. HTTP Connectivity & Model Status
Write-Host "[2/4] HTTP Connectivity" -ForegroundColor Yellow
if ($env:RUNPOD_BASE_URL) {
    $Models_URL = "$($env:RUNPOD_BASE_URL)/models"
    try {
        $start = Get-Date
        $Response = Invoke-RestMethod -Uri $Models_URL -Method Get -TimeoutSec 5
        $end = Get-Date
        $latency = ($end - $start).TotalMilliseconds
        Write-Host "  /v1/models      : OK ($($latency)ms)" -ForegroundColor Green
        
        if ($Response.data) {
            $Models = $Response.data | ForEach-Object { $_.id }
            Write-Host "  Models Found    : $($Models -join ', ')"
            if ($Models -contains $env:MODEL_ID) {
                Write-Host "  Target Model    : PRESENT" -ForegroundColor Green
            } else {
                Write-Host "  Target Model    : MISSING ($($env:MODEL_ID))" -ForegroundColor Red
            }
        }
    } catch {
        Write-Host "  /v1/models      : FAILED ($($_.Exception.Message))" -ForegroundColor Red
    }
} else {
    Write-Host "  RUNPOD_BASE_URL is not set." -ForegroundColor Gray
}

# 3. SSH Connectivity (only if needed for generic mode)
Write-Host "[3/4] SSH Connectivity" -ForegroundColor Yellow
$SSH_KEY = $env:RUNPOD_SSH_KEY
$REMOTE_HOST = $env:RUNPOD_SSH_HOST
$SSH_PORT = $env:RUNPOD_SSH_PORT

if ($SSH_KEY -and $REMOTE_HOST) {
    Write-Host "  Direct SSH ($REMOTE_HOST)..." -NoNewline
    ssh -i $SSH_KEY -p $SSH_PORT -o ConnectTimeout=5 -o StrictHostKeyChecking=no $REMOTE_HOST "echo SSH_OK" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
        
        Write-Host "  SCP Support     ..." -NoNewline
        $TempFile = [System.IO.Path]::GetTempFileName()
        "SCP_TEST" | Out-File -FilePath $TempFile -Encoding ASCII
        scp -i $SSH_KEY -P $SSH_PORT -o ConnectTimeout=5 -o StrictHostKeyChecking=no $TempFile "${REMOTE_HOST}:/tmp/scp_test.tmp" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host " OK" -ForegroundColor Green
            ssh -i $SSH_KEY -p $SSH_PORT $REMOTE_HOST "rm /tmp/scp_test.tmp"
        } else {
            Write-Host " FAILED" -ForegroundColor Yellow
        }
        Remove-Item $TempFile
    } else {
        Write-Host " FAILED" -ForegroundColor Red
    }
} else {
    Write-Host "  SSH details not fully provided in env (Generic mode may fail)." -ForegroundColor Gray
}

# 4. Managed Pod Heuristics
Write-Host "[4/4] Remote Environment Checks" -ForegroundColor Yellow
if ($SSH_KEY -and $REMOTE_HOST) {
    $HEURISTIC_CMD = "pgrep -f 'vllm serve' >/dev/null && echo 'VLLM_RUNNING' || echo 'VLLM_NOT_RUNNING'"
    $Result = ssh -i $SSH_KEY -p $SSH_PORT -o ConnectTimeout=5 $REMOTE_HOST $HEURISTIC_CMD 2>$null
    if ($Result -match "VLLM_RUNNING") {
        Write-Host "  vLLM Process    : DETECTED (Likely a managed service pod)" -ForegroundColor Yellow
        Write-Host "  Recommendation  : Use RUNPOD_MODE=existing"
    } else {
        Write-Host "  vLLM Process    : NOT FOUND (Safe for generic deployment)" -ForegroundColor Green
    }
} else {
    Write-Host "  Skipping remote checks (no SSH access)." -ForegroundColor Gray
}

Write-Host "--- Diagnostics Complete ---" -ForegroundColor Cyan
