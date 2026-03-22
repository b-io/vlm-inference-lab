# demo_end_to_end.ps1 - Orchestrate the full Azure GPU demo (Optional/Archival)
# Note: Runpod is now the recommended primary demo path.

# Support UTF-8 emojis in PowerShell console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Configuration
$TF_DIR = "infra\terraform\azure\single-node"
$MODEL = if ($args[0]) { $args[0] } else { "facebook/opt-125m" } # Default lightweight model

Write-Host "--------------------------------------------------------"
Write-Host "VLM Inference Lab - End-to-End Demo (Azure)"
Write-Host "--------------------------------------------------------"

# 1. Terraform Plan & Apply
Write-Host "Provisioning Azure Infrastructure..."
Push-Location $TF_DIR
terraform init
terraform apply -auto-approve
Pop-Location

$VM_IP = Push-Location $TF_DIR; terraform output -raw public_ip; Pop-Location
$SSH_CMD = Push-Location $TF_DIR; terraform output -raw ssh_command; Pop-Location

Write-Host "VM Provisioned at: $VM_IP"

# 2. Start vLLM on remote VM
Write-Host "Starting vLLM on Remote VM..."
# We assume standard SSH is available and key is configured
$HF_TOKEN_VAL = if ($env:HF_TOKEN) { $env:HF_TOKEN } else { "" }
ssh -o StrictHostKeyChecking=no $VM_IP "cd vlm-inference-lab && HF_TOKEN='$HF_TOKEN_VAL' ./scripts/start_vllm_remote.sh '$MODEL'"

# 3. Wait for Healthcheck
Write-Host "Waiting for vLLM to be healthy (this may take a few minutes for model download)..."
$MAX_RETRIES = 30
$RETRY_COUNT = 0

while ($true) {
    try {
        $response = Invoke-WebRequest -Uri "http://${VM_IP}:8000/v1/models" -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "vLLM is Healthy!"
            break
        }
    } catch {
        # Ignore errors during wait
    }

    $RETRY_COUNT++
    if ($RETRY_COUNT -ge $MAX_RETRIES) {
        Write-Error "Error: vLLM did not become healthy in time."
        exit 1
    }
    
    $WaitMsg = "    Attempt $RETRY_COUNT/$MAX_RETRIES: vLLM is not ready yet..."
    Write-Host "    $WaitMsg"
    Start-Sleep -Seconds 20
}

# 4. Run Local Benchmark
Write-Host "Running Local Benchmark against remote VM..."
.\scripts\benchmark_remote.ps1 "http://${VM_IP}:8000/v1" "$MODEL"

Write-Host "--------------------------------------------------------"
Write-Host "Demo Complete"
Write-Host "--------------------------------------------------------"
Write-Host "Results saved in results/ directory."
Write-Host "To tear down infrastructure, run: cd $TF_DIR; terraform destroy"
Write-Host "--------------------------------------------------------"
