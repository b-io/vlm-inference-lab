# scripts/runpod/demo_end_to_end.ps1
# Refactored Runpod Orchestrator supporting 'existing' and 'generic' modes.
# Now with SSH bootstrapping, managed-instance detection, and SCP fallback.

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

# 2. Set Defaults and Validate Configuration
$MODE = $env:RUNPOD_MODE
if (-not $MODE) { $MODE = "existing" }

if (-not $env:RUNPOD_BASE_URL) { Write-Error "RUNPOD_BASE_URL is required for both modes"; exit 1 }
if (-not $env:MODEL_ID) { Write-Error "MODEL_ID is required"; exit 1 }

# New Configuration Variables
$SSH_STRATEGY = if ($env:RUNPOD_SSH_MODE) { $env:RUNPOD_SSH_MODE } else { "auto" }
$PROXY_SSH_TARGET = $env:RUNPOD_PROXY_SSH_TARGET
$ALLOW_MUTATION = if ($env:RUNPOD_ALLOW_MANAGED_INSTANCE_MUTATION) { [System.Convert]::ToBoolean($env:RUNPOD_ALLOW_MANAGED_INSTANCE_MUTATION) } else { $false }

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
    
    if ($url -notlike "*.proxy.runpod.net*" -and $url -notlike "*localhost*" -and $url -notlike "*127.0.0.1*") {
        Write-Warning "RUNPOD_BASE_URL ($url) does not appear to be a standard Runpod proxy URL (*.proxy.runpod.net)."
    }
}

# Helper to show SSH troubleshooting
function Show-SshTroubleshooting($Message, $SshKey, $Strategy) {
    Write-Host ""
    Write-Host $Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting Runpod SSH (Strategy: $Strategy):"
    Write-Host "1. 'generic' mode requires DIRECT TCP SSH access for SCP/SFTP support."
    Write-Host "2. The Runpod 'ssh.runpod.io' shortcut (Proxied SSH) IS NOT SUFFICIENT for SCP/SFTP."
    Write-Host "3. If direct TCP SSH failed, you may need to bootstrap 'sshd' via Proxied SSH."
    Write-Host "4. Use 'RUNPOD_PROXY_SSH_TARGET' in .env.runpod (e.g., j0axhra8c4u1mi-64411b50@ssh.runpod.io)."
    Write-Host "5. Ensure direct TCP host/port are correct from the Connect tab ('SSH over exposed TCP')."
    Write-Host "6. Ensure your SSH key ($SshKey) is correctly configured."
    Write-Host ""
    Write-Host "If you already have a pod running vLLM, use 'RUNPOD_MODE=existing' instead."
}

# Helper to detect managed instance
function Get-IsManagedInstance($baseUrl, $remoteTarget, $sshKey, $sshPort) {
    # Check 1: HTTP Readiness
    $readinessUrl = "$baseUrl/models"
    try {
        $resp = Invoke-RestMethod -Uri $readinessUrl -Method Get -TimeoutSec 5
        if ($resp.data) { return $true }
    } catch {}

    # Check 2: Remote Process Check
    if ($remoteTarget) {
        $checkCmd = "pgrep -f 'vllm serve' || echo 'NOT_FOUND'"
        $out = ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i $sshKey -p $sshPort $remoteTarget $checkCmd
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
    if (-not $env:RUNPOD_SSH_HOST) { Write-Error "RUNPOD_SSH_HOST is required for generic mode"; exit 1 }
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
    
    $DIRECT_SSH_OK = $false
    $SCP_OK = $false

    # Step 4.1: Try Direct SSH Preflight
    Write-Host "Direct SSH Preflight Check: Connecting to $REMOTE_HOST (Port: $SSH_PORT)..."
    ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i $SSH_KEY -p $SSH_PORT $REMOTE_HOST "echo SSH_OK"
    if ($LASTEXITCODE -eq 0) {
        $DIRECT_SSH_OK = $true
        Write-Host "Direct SSH connection verified."
    }

    # Step 4.2: Try Bootstrap if Direct failed and strategy allows
    if (-not $DIRECT_SSH_OK -and ($SSH_STRATEGY -eq "auto" -or $SSH_STRATEGY -eq "proxied")) {
        if (-not $PROXY_SSH_TARGET) {
            Write-Host "Direct SSH failed and RUNPOD_PROXY_SSH_TARGET is not configured. Cannot bootstrap." -ForegroundColor Yellow
        } else {
            Write-Host "Attempting SSH Bootstrap via Proxied SSH ($PROXY_SSH_TARGET)..."
            $Pub_Key = ""
            $Pub_Key_Path = "$SSH_KEY.pub"
            if (Test-Path $Pub_Key_Path) { $Pub_Key = Get-Content $Pub_Key_Path -Raw }
            
            $Bootstrap_Script = Get-Content "scripts/runpod/enable_sshd.sh" -Raw
            # We use a heredoc-like approach to send the script over proxied SSH
            $Encoded_Script = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($Bootstrap_Script))
            $Remote_Init = "echo '$Encoded_Script' | base64 -d > /tmp/bootstrap.sh && bash /tmp/bootstrap.sh '$Pub_Key'"
            
            ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no $PROXY_SSH_TARGET $Remote_Init
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Bootstrap successful. Retrying direct SSH..."
                ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i $SSH_KEY -p $SSH_PORT $REMOTE_HOST "echo SSH_OK"
                if ($LASTEXITCODE -eq 0) { $DIRECT_SSH_OK = $true; Write-Host "Direct SSH now verified." }
            } else {
                Write-Host "Bootstrap via proxied SSH failed." -ForegroundColor Red
            }
        }
    }

    if (-not $DIRECT_SSH_OK) {
        Show-SshTroubleshooting "Error: SSH connectivity failed to $REMOTE_HOST" $SSH_KEY $SSH_STRATEGY
        exit 1
    }

    # Step 4.3: SCP Preflight
    Write-Host "SCP Preflight Check..."
    $TempFile = [System.IO.Path]::GetTempFileName()
    "SCP_TEST" | Out-File -FilePath $TempFile -Encoding ASCII
    $RemoteTemp = "~/scp_test_$(Get-Random).tmp"
    
    scp -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i $SSH_KEY -P $SSH_PORT $TempFile "${REMOTE_HOST}:${RemoteTemp}"
    if ($LASTEXITCODE -eq 0) {
        $SCP_OK = $true
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no -i $SSH_KEY -p $SSH_PORT $REMOTE_HOST "rm -f $RemoteTemp"
        Write-Host "SCP transfer verified."
    } else {
        Write-Host "SCP failed. Falling back to SSH-only remote file creation." -ForegroundColor Yellow
    }
    Remove-Item $TempFile -ErrorAction SilentlyContinue

    # Step 4.4: Deployment
    Write-Host "Creating remote directory..."
    ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $SSH_PORT $REMOTE_HOST "mkdir -p ~/vlm-inference-lab/scripts/runpod"
    
    Write-Host "Deploying startup script..."
    if ($SCP_OK) {
        scp -o StrictHostKeyChecking=no -i $SSH_KEY -P $SSH_PORT scripts/runpod/start_vllm.sh "${REMOTE_HOST}:~/vlm-inference-lab/scripts/runpod/"
    } else {
        # Fallback: Create script via SSH heredoc
        $Script_Content = Get-Content "scripts/runpod/start_vllm.sh" -Raw
        $Encoded_Script = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($Script_Content))
        ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $SSH_PORT $REMOTE_HOST "echo '$Encoded_Script' | base64 -d > ~/vlm-inference-lab/scripts/runpod/start_vllm.sh && chmod +x ~/vlm-inference-lab/scripts/runpod/start_vllm.sh"
    }
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to deploy startup script"; exit 1 }

    Write-Host "Starting vLLM remotely..."
    $SHORT_MODEL = $env:MODEL_ID.Split('/')[-1]
    $ALLOW_KILL = if ($ALLOW_MUTATION) { "true" } else { "false" }
    $START_CMD = "bash ~/vlm-inference-lab/scripts/runpod/start_vllm.sh '$($env:MODEL_ID)' '$($env:HF_TOKEN)' $($env:VLLM_PORT) '$SHORT_MODEL' $ALLOW_KILL"
    ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $SSH_PORT $REMOTE_HOST $START_CMD
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
Write-Host "Starting benchmark using $env:RUNPOD_BASE_URL (Model: $RESOLVED_MODEL_ID)..."
.\scripts\benchmark_remote.ps1 $env:RUNPOD_BASE_URL $RESOLVED_MODEL_ID 10 2 --fail-on-errors
if ($LASTEXITCODE -ne 0) { Write-Error "Benchmark failed"; exit 1 }

Write-Host "--------------------------------------------------------"
Write-Host "Orchestration Successful!"
Write-Host "--------------------------------------------------------"
Write-Host "IMPORTANT: Don't forget to TERMINATE your Runpod pod"
Write-Host "--------------------------------------------------------"
