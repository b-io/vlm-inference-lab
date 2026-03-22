# scripts/runpod/demo_end_to_end.ps1
# Refactored Runpod Orchestrator supporting 'existing' and 'generic' modes.
# Supports Proxied SSH fallback, managed-instance detection, and SCP fallback.

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

# 2. Set Defaults and Validate Configuration
$MODE = $env:RUNPOD_MODE
if (-not $MODE) { $MODE = "existing" }

if (-not $env:RUNPOD_BASE_URL) { Write-Error "RUNPOD_BASE_URL is required for both modes"; exit 1 }
if (-not $env:MODEL_ID) { Write-Error "MODEL_ID is required"; exit 1 }

# New Configuration Variables
$SSH_STRATEGY = if ($env:RUNPOD_SSH_MODE) { $env:RUNPOD_SSH_MODE } else { "auto" }
$PROXY_SSH_TARGET = $env:RUNPOD_PROXY_SSH_TARGET
$ALLOW_MUTATION = if ($env:RUNPOD_ALLOW_MANAGED_INSTANCE_MUTATION -eq "true") { $true } else { $false }

# Helper to validate RUNPOD_BASE_URL
function Assert-ValidRunpodBaseUrl($url, $mode) {
    if ($url -like "*127.0.0.1*" -or $url -like "*localhost*") {
        return
    }
    
    if ($mode -eq "existing") {
        if ($url -match "^http://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+") {
            Write-Error "In 'existing' mode, RUNPOD_BASE_URL must point to the HTTP service endpoint, typically 'https://<pod-id>-8000.proxy.runpod.net/v1'. Do not use the raw SSH/public IP for HTTP readiness checks."
            exit 1
        }
    }
}

# Helper to show SSH troubleshooting
function Show-SshTroubleshooting($Message, $SshKey, $Strategy) {
    Write-Host ""
    Write-Host $Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting Runpod SSH (Strategy: $Strategy):"
    Write-Host "1. 'generic' mode requires shell access for deployment."
    Write-Host "2. Proxied SSH (ssh.runpod.io) is supported even if direct TCP SSH is blocked."
    Write-Host "3. Ensure 'RUNPOD_PROXY_SSH_TARGET' is set in .env.runpod (e.g. user@ssh.runpod.io)."
    Write-Host "4. If direct TCP failed, ensure your SSH key ($SshKey) is correctly configured."
    Write-Host "5. If you already have a pod running vLLM, use 'RUNPOD_MODE=existing' instead."
    Write-Host ""
}

# Helper to detect managed instance
function Get-IsManagedInstance($baseUrl, $remoteTarget, $sshKey, $sshPort) {
    # Check 1: HTTP Readiness
    $readinessUrl = "$baseUrl/models"
    try {
        $resp = Invoke-RestMethod -Uri $readinessUrl -Method Get -TimeoutSec 5
        if ($resp.data) { return $true }
    } catch {}

    # Check 2: Remote Process Check (if direct SSH works)
    if ($remoteTarget -and $sshKey) {
        $checkCmd = "pgrep -f 'vllm serve' || echo 'NOT_FOUND'"
        $out = ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i $sshKey -p $sshPort $remoteTarget $checkCmd 2>$null
        if ($out -ne "NOT_FOUND" -and $out -match "\d+") { return $true }
    }
    
    return $false
}

Assert-ValidRunpodBaseUrl $env:RUNPOD_BASE_URL $MODE

Write-Host "--------------------------------------------------------"
Write-Host "VLM Inference Lab: Runpod Orchestrator"
Write-Host "Mode        : $MODE"
Write-Host "Model ID    : $($env:MODEL_ID)"
Write-Host "Base URL    : $($env:RUNPOD_BASE_URL)"
Write-Host "SSH Strategy: $SSH_STRATEGY"
Write-Host "--------------------------------------------------------"

if ($MODE -eq "generic") {
    if (-not $env:RUNPOD_SSH_HOST -and $SSH_STRATEGY -ne "proxied") { 
        Write-Error "RUNPOD_SSH_HOST is required for generic mode unless RUNPOD_SSH_MODE=proxied"; exit 1 
    }
    if (-not $env:RUNPOD_SSH_USER) { $env:RUNPOD_SSH_USER = "root" }
    if (-not $env:RUNPOD_SSH_PORT) { $env:RUNPOD_SSH_PORT = "22" }
    if (-not $env:RUNPOD_SSH_KEY_PATH) { $env:RUNPOD_SSH_KEY_PATH = "~/.ssh/id_ed25519" }
}

# 3. Detection & Guard
$REMOTE_HOST = "$($env:RUNPOD_SSH_USER)@$($env:RUNPOD_SSH_HOST)"
$SSH_KEY = $env:RUNPOD_SSH_KEY_PATH
$SSH_PORT = $env:RUNPOD_SSH_PORT

Write-Host "Stage: Environment Detection"
$IS_MANAGED = Get-IsManagedInstance $env:RUNPOD_BASE_URL $REMOTE_HOST $SSH_KEY $SSH_PORT
if ($IS_MANAGED) {
    Write-Host "Detected: This host appears to be an already-managed vLLM service pod." -ForegroundColor Yellow
    if ($MODE -eq "generic" -and -not $ALLOW_MUTATION) {
        Write-Host "Error: Generic remote deployment is disabled by default on managed pods to avoid killing the main service process." -ForegroundColor Red
        Write-Host "Suggestion: Use 'RUNPOD_MODE=existing', or set 'RUNPOD_ALLOW_MANAGED_INSTANCE_MUTATION=true' only if you are sure."
        exit 1
    }
} else {
    Write-Host "Detected: Generic host (not a managed vLLM service)."
}

# 4. Generic Mode: SSH Deployment
if ($MODE -eq "generic") {
    Write-Host "--- Stage: Remote SSH Deployment ---"
    
    $DEPLOY_TARGET = $null
    $DEPLOY_SSH_OPTS = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no")
    $USE_SCP = $false
    $DIAGNOSTICS = @{
        "Direct SSH"  = "unavailable"
        "Proxied SSH" = "unavailable"
        "SCP Support" = "unavailable"
        "Fallback Mode" = "none"
    }

    # Decide on transport
    if ($SSH_STRATEGY -eq "proxied") {
        if (-not $PROXY_SSH_TARGET) { Write-Error "RUNPOD_PROXY_SSH_TARGET is required for proxied strategy"; exit 1 }
        $DEPLOY_TARGET = $PROXY_SSH_TARGET
        $DIAGNOSTICS["Proxied SSH"] = "available"
        Write-Host "Using Proxied SSH transport: $DEPLOY_TARGET"
    } else {
        # Try Direct first
        Write-Host "Attempting Direct SSH Preflight ($REMOTE_HOST)..."
        ssh @DEPLOY_SSH_OPTS -i $SSH_KEY -p $SSH_PORT $REMOTE_HOST "echo SSH_OK" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $DEPLOY_TARGET = $REMOTE_HOST
            $DEPLOY_SSH_OPTS += @("-i", $SSH_KEY, "-p", $SSH_PORT)
            $DIAGNOSTICS["Direct SSH"] = "available"
            Write-Host "Direct SSH verified."
            
            # Try SCP preflight if direct SSH works
            $TempFile = [System.IO.Path]::GetTempFileName()
            "SCP_TEST" | Out-File -FilePath $TempFile -Encoding ASCII
            scp @DEPLOY_SSH_OPTS -P $SSH_PORT $TempFile "${REMOTE_HOST}:/tmp/scp_test.tmp" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $USE_SCP = $true
                $DIAGNOSTICS["SCP Support"] = "available"
                ssh @DEPLOY_SSH_OPTS $REMOTE_HOST "rm /tmp/scp_test.tmp"
                Write-Host "SCP transfer verified."
            } else {
                $DIAGNOSTICS["Fallback Mode"] = "SSH-only file creation"
                Write-Host "SCP failed. Will use SSH-only fallback." -ForegroundColor Yellow
            }
            Remove-Item $TempFile
        } elseif ($SSH_STRATEGY -eq "auto" -and $PROXY_SSH_TARGET) {
            Write-Host "Direct SSH failed. Attempting Proxied SSH fallback ($PROXY_SSH_TARGET)..."
            ssh @DEPLOY_SSH_OPTS $PROXY_SSH_TARGET "echo SSH_OK" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $DEPLOY_TARGET = $PROXY_SSH_TARGET
                $DIAGNOSTICS["Proxied SSH"] = "available"
                $DIAGNOSTICS["Fallback Mode"] = "SSH-only file creation"
                Write-Host "Proxied SSH verified."
            }
        }
    }

    Write-Host "`nConnection Summary:"
    $DIAGNOSTICS.GetEnumerator() | ForEach-Object { Write-Host "  $($_.Key): $($_.Value)" }
    Write-Host ""

    if (-not $DEPLOY_TARGET) {
        Show-SshTroubleshooting "Error: SSH connectivity failed." $SSH_KEY $SSH_STRATEGY
        exit 1
    }

    # Step 4.2: Deploy startup script
    Write-Host "Creating remote directory..."
    ssh @DEPLOY_SSH_OPTS $DEPLOY_TARGET "mkdir -p ~/vlm-inference-lab/scripts/runpod"

    Write-Host "Deploying startup script..."
    $Script_Content = Get-Content "scripts/runpod/start_vllm.sh" -Raw
    if ($USE_SCP) {
        scp @DEPLOY_SSH_OPTS scripts/runpod/start_vllm.sh "${DEPLOY_TARGET}:~/vlm-inference-lab/scripts/runpod/start_vllm.sh"
    } else {
        # SSH fallback: Use Base64 to avoid escaping issues
        $Encoded_Script = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($Script_Content))
        ssh @DEPLOY_SSH_OPTS $DEPLOY_TARGET "echo '$Encoded_Script' | base64 -d > ~/vlm-inference-lab/scripts/runpod/start_vllm.sh && chmod +x ~/vlm-inference-lab/scripts/runpod/start_vllm.sh"
    }
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to deploy startup script"; exit 1 }

    Write-Host "Starting vLLM remotely..."
    $ALLOW_KILL = if ($env:RUNPOD_ALLOW_KILL_EXISTING_VLLM -eq "true") { "true" } else { "false" }
    $START_CMD = "bash ~/vlm-inference-lab/scripts/runpod/start_vllm.sh '$($env:MODEL_ID)' '$($env:HF_TOKEN)' $($env:VLLM_PORT) '' $ALLOW_KILL"
    ssh @DEPLOY_SSH_OPTS $DEPLOY_TARGET $START_CMD
    if ($LASTEXITCODE -ne 0) { Write-Error "Remote startup script failed"; exit 1 }
}

# 5. Readiness Check (Common for both modes)
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
        }
    } catch {
        $StatusCode = ""
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode) { $StatusCode = " [HTTP $($_.Exception.Response.StatusCode.value__)]" }
        Write-Host "Waiting for endpoint to respond...$StatusCode ($($_.Exception.Message))"
    }
    $RETRY_COUNT++
    Start-Sleep -Seconds 10
}

if (-not $HEALTHY) {
    Write-Host ""
    Write-Host "Timeout: vLLM or model failed to become healthy at $READINESS_URL" -ForegroundColor Red
    Write-Host "Mode        : $MODE"
    Write-Host "Base URL    : $($env:RUNPOD_BASE_URL)"
    Write-Host "Model ID    : $($env:MODEL_ID)"
    Write-Host "Suggestion  : Check your Runpod pod logs for startup errors or OOM."
    exit 1
}

# 6. Remote Benchmark
Write-Host "--- Stage: Remote Benchmark ---"
$TIER = if ($env:BENCHMARK_TIER) { $env:BENCHMARK_TIER } else { "smoke" }
    if ($env:BENCHMARK_PATH -eq "vllm") {
        Write-Host "Starting professional benchmark (vllm bench) using $env:RUNPOD_BASE_URL (Model: $RESOLVED_MODEL_ID)..."
        .\scripts\benchmark_vllm_bench.ps1 $env:RUNPOD_BASE_URL $RESOLVED_MODEL_ID $TIER
    } elseif ($env:BENCHMARK_PATH -eq "sweep") {
        Write-Host "Starting professional sweep (vllm bench sweep) using $env:RUNPOD_BASE_URL (Model: $RESOLVED_MODEL_ID)..."
        .\scripts\benchmark_vllm_sweep.ps1 $env:RUNPOD_BASE_URL $RESOLVED_MODEL_ID $TIER
    } else {
        Write-Host "Starting benchmark ($TIER) using $env:RUNPOD_BASE_URL (Model: $RESOLVED_MODEL_ID)..."
        .\scripts\benchmark_remote.ps1 $env:RUNPOD_BASE_URL $RESOLVED_MODEL_ID --tier $TIER --fail-on-errors
    }
if ($LASTEXITCODE -ne 0) { Write-Error "Benchmark failed"; exit 1 }

Write-Host "--------------------------------------------------------"
Write-Host "Orchestration Successful!"
Write-Host "--------------------------------------------------------"
Write-Host "Tier        : $(if ($env:BENCHMARK_TIER) { $env:BENCHMARK_TIER } else { 'smoke' })"
Write-Host "IMPORTANT: Don't forget to TERMINATE your Runpod pod"
Write-Host "--------------------------------------------------------"
