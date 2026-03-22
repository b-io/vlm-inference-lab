# Runpod GPU Serving Demo

This guide explains how to run a real-world VLM inference benchmark using **Runpod** as the GPU provider and **vLLM** as the serving engine.

## 🕒 Estimated Setup Time: 5–10 minutes

## 1. Prerequisites

- A **Runpod account** with credits (billing set up).
- **SSH keys** added to your Runpod account.
- **Python 3.11+** installed locally.
- **Hugging Face Token** (required if serving gated models like Llama).

## 2. Launching the GPU Pod (Manual Mode)

The most reliable way to start is using the Runpod UI:

1. **Go to [Runpod Console](https://www.runpod.io/console/pods).**
2. **Select a GPU**: We recommend **NVIDIA H100 PCIe** or **NVIDIA A100-SXM4-80GB**.
3. **Choose Template**: Select the **vLLM** template if available, or a generic **PyTorch** template (e.g., `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel`).
4. **Storage**: Set Container Disk to at least **20GB** and Volume Disk to at least **50GB** (to store model weights).
5. **Expose Ports**: Ensure port **8000** (HTTP) and **22** (SSH) are exposed.
6. **Deploy**: Wait for the pod to become `Running`.

## 3. Local Environment Setup

1. **Clone the repo** (if you haven't already):
   ```bash
   git clone https://github.com/<your-username>/vlm-inference-lab.git
   cd vlm-inference-lab
   ```

2. **Install local dependencies**:
   ```bash
   pip install -e .
   ```

3. **Create `.env.runpod`**:
   ```bash
   cp .env.runpod.example .env.runpod
   ```
   Follow the comments in the file to configure `RUNPOD_MODE`, `RUNPOD_BASE_URL` (the HTTP source of truth), and `RUNPOD_SSH_*` variables (for generic remote deployment).

## 4. Run the End-to-End Demo

The orchestrator script supports two modes based on your `.env.runpod` configuration:

### Mode A: Existing Runpod vLLM Pod (Recommended)
Use this if you launched a pod using a **vLLM template** that is already running.

1. Set `RUNPOD_MODE=existing` in `.env.runpod`.
2. Provide `RUNPOD_BASE_URL` (e.g., `https://<pod-id>-8000.proxy.runpod.net/v1`).
3. Set `MODEL_ID` (e.g., `facebook/opt-125m`).
4. Run:
   ```bash
   ./scripts/demo_runpod_end_to_end.sh
   ```

### Mode B: Deploy from Local Machine over SSH (Generic Remote)
Use this to deploy vLLM from your local machine to a **generic GPU VM/pod** (e.g., a PyTorch template).

1. Set `RUNPOD_MODE=generic` in `.env.runpod`.
2. Provide `RUNPOD_SSH_HOST`, `RUNPOD_SSH_PORT`, `RUNPOD_SSH_USER`, and `RUNPOD_SSH_KEY_PATH`.
3. Provide `RUNPOD_BASE_URL` (e.g., `http://<pod-ip>:8000/v1`).
4. Set `MODEL_ID` (e.g., `facebook/opt-125m`).
5. Run:
   ```bash
   ./scripts/demo_runpod_end_to_end.sh
   ```

The script will:
1. (Generic mode only) SSH into the pod and upload/start vLLM.
2. Wait for the model to load and the health check to pass via the proxy or direct URL.
3. Run a benchmark from your local machine against the remote endpoint.

## 5. Manual CLI Operations

If you prefer to run steps individually:

### Start vLLM on the Remote Pod
```bash
# On your local machine
ssh -p <ssh-port> root@<ssh-host> "mkdir -p ~/vlm-inference-lab/scripts/cloud"
scp -P <ssh-port> scripts/cloud/runpod_start_vllm.sh root@<ssh-host>:~/vlm-inference-lab/scripts/cloud/
ssh -p <ssh-port> root@<ssh-host> "bash ~/vlm-inference-lab/scripts/cloud/runpod_start_vllm.sh facebook/opt-125m"
```

### Run Benchmark Locally
```bash
./scripts/benchmark_remote.sh http://<pod-ip>:8000/v1 facebook/opt-125m
```

## 6. Cleanup

**CRITICAL**: Runpod charges by the hour. When you are finished with the demo:
1. Go to the Runpod Console.
2. **Terminate** the pod (do not just stop it, or you will still be charged for storage).

---

## 💡 Troubleshooting

- **SSH Connection Refused**: Ensure you added your public key to Runpod and the pod is fully started.
- **CUDA Out of Memory**: Choose a smaller model (e.g., `facebook/opt-125m`) or a larger GPU (80GB A100).
- **vLLM Container Fails**: Check the logs on the pod: `docker logs vllm-server`.
