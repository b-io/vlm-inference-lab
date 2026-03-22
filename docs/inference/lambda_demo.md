# Lambda Labs GPU Serving Demo

This guide explains how to run a real-world VLM inference benchmark using **Lambda Labs** as the GPU provider and **vLLM** as the serving engine.

## 🕒 Estimated Setup Time: 5–10 minutes

## 1. Prerequisites

- A **Lambda Labs account** with credits.
- **SSH keys** added to your Lambda account.
- **Python 3.11+** installed locally.

## 2. Launching the GPU Instance

1. **Go to [Lambda Console](https://cloud.lambdalabs.com/instances).**
2. **Select Instance Type**: Choose `gpu_1x_h100` (NVIDIA H100) or `gpu_1x_a100_80gb` (NVIDIA A100).
3. **Region**: Choose a region with available capacity.
4. **SSH Key**: Select your public key.
5. **Launch**: Wait for the instance to transition to `Running`.

## 3. Local Environment Setup

1. **Clone the repo** (if you haven't already):
   ```bash
   git clone https://github.com/<your-username>/vlm-inference-lab.git
   cd vlm-inference-lab
   ```

2. **Create `.env.lambda`**:
   ```bash
   cp .env.lambda.example .env.lambda
   ```
   Fill in the `LAMBDA_INSTANCE_IP` and `HF_TOKEN`.

## 4. Run the End-to-End Demo

The orchestrator script for Lambda works similarly to the Runpod one, handling remote startup and local benchmarking.

```bash
./scripts/lambda/demo_end_to_end.sh
```

## 5. Manual Setup on Lambda Instance

If you prefer to run steps manually on the remote instance:

1. **SSH into the instance**:
   ```bash
   ssh ubuntu@<instance-ip>
   ```

2. **Check for Docker**:
   Lambda instances usually come with Docker and NVIDIA drivers pre-installed. Verify with:
   ```bash
   docker ps
   nvidia-smi
   ```

3. **Start vLLM**:
   You can use the same startup script as Runpod, ensuring port 8000 is open.

## 6. Cleanup

**CRITICAL**: Lambda charges by the hour. When finished:
1. Go to the Lambda Console.
2. **Terminate** the instance immediately.

---

## 💡 Troubleshooting

- **No Capacity**: Lambda H100s/A100s are often in high demand. If none are available, try another region or check **Runpod** as our primary provider.
- **Firewall**: Ensure your local IP has access to port 8000 and 22 if you configured a security group.
