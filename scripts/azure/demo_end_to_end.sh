#!/bin/bash
# demo_end_to_end.sh - Orchestrate the full Azure GPU demo (Optional/Archival)
# Note: Runpod is now the recommended primary demo path.

set -e

# Configuration
TF_DIR="infra/terraform/azure/single-node"
MODEL="facebook/opt-125m" # Default lightweight model

echo "--------------------------------------------------------"
echo "VLM Inference Lab - End-to-End Demo (Azure)"
echo "--------------------------------------------------------"

# 1. Terraform Plan & Apply
echo "Provisioning Azure Infrastructure..."
cd $TF_DIR
terraform init
terraform apply -auto-approve
cd -

VM_IP=$(cd $TF_DIR && terraform output -raw public_ip)
SSH_CMD=$(cd $TF_DIR && terraform output -raw ssh_command)
echo "VM Provisioned at: $VM_IP"

# 2. Start vLLM on remote VM
echo "Starting vLLM on Remote VM..."
ssh -o StrictHostKeyChecking=no $VM_IP "cd vlm-inference-lab && HF_TOKEN=$HF_TOKEN ./scripts/azure/start_vllm.sh $MODEL"

# 3. Wait for Healthcheck
echo "Waiting for vLLM to be healthy (this may take a few minutes for model download)..."
MAX_RETRIES=30
RETRY_COUNT=0
until curl -s http://$VM_IP:8000/v1/models > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Error: vLLM did not become healthy in time."
        exit 1
    fi
    echo "    Attempt $RETRY_COUNT/$MAX_RETRIES: vLLM is not ready yet..."
    sleep 20
done
echo "vLLM is Healthy!"

# 4. Run Local Benchmark
echo "Running Local Benchmark against remote VM..."
./scripts/benchmark_remote.sh "http://$VM_IP:8000/v1" "$MODEL"

echo "--------------------------------------------------------"
echo "Demo Complete"
echo "--------------------------------------------------------"
echo "Results saved in results/ directory."
echo "To tear down infrastructure, run: cd $TF_DIR && terraform destroy"
echo "--------------------------------------------------------"
