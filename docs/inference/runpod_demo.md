# Runpod GPU Serving Demo

This guide explains how to run a real-world VLM inference benchmark using **Runpod** as the GPU provider and **vLLM** as
the serving engine.

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
3. **Choose Template**: Select the **vLLM** template if available, or a generic **PyTorch** template (e.g.,
   `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel`).
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
   Follow the comments in the file to configure `RUNPOD_MODE`, `RUNPOD_BASE_URL` (the HTTP source of truth), and
   `RUNPOD_SSH_*` variables (for generic remote deployment).

## 4. Which mode should I use?

### Use `existing` when:

- The pod is already serving a model (e.g., via a Runpod vLLM template).
- You only want to benchmark or inspect the performance of an active endpoint.
- No SSH access is required for this mode.

### Use `generic` when:

- You want to deploy a model remotely from your local machine to a generic GPU host.
- The pod is a standard SSH-managed host (e.g., a PyTorch or Ubuntu template).
- You need the script to manage the lifecycle of the vLLM process/container.

### Connection Strategy

- **HTTP Benchmark/Readiness**: Always use the Runpod proxy base URL (`https://<pod-id>-8000.proxy.runpod.net/v1`). The
  professional vLLM benchmark wrappers automatically normalize a trailing `/v1` from `RUNPOD_BASE_URL` and pass explicit
  endpoints (e.g., `/v1/completions`).
- **Proxied SSH**: Sufficient for shell access and remote file creation if port 22 is blocked.
- **Direct TCP SSH**: Preferred for faster SCP/SFTP transfers if available.

## 5. Benchmark Tiers

The orchestrator and benchmark scripts support **tiers** to provide sensible defaults for different scenarios:

| Tier         | Purpose                                          | Requests | Concurrency |
|:-------------|:-------------------------------------------------|:---------|:------------|
| `smoke`      | Verify endpoint, auth, and basic model response. | 10       | 1           |
| `latency`    | Characterize stable latency percentiles.         | 100      | 1           |
| `throughput` | Compare GPU/model/server capacity under load.    | 200      | 8           |
| `sweep`      | Parameter grid search for Pareto analysis.       | N/A      | Grid        |

> [!NOTE]
> A **smoke test** (10 requests) is useful for verification but is **not** a serious GPU benchmark. Use `latency`,
`throughput`, or `sweep` for credible performance data.

### Professional Benchmarking & Pareto Analysis

For serious characterization, this repo orchestrates **vLLM's native benchmark tools** (`vllm bench serve` and
`vllm bench sweep serve`).

**Note on Stability:**

- `vllm bench serve` is the **stable, validated path** for remote benchmarking in this repo.
- `vllm bench sweep serve` is supported but should be treated as **version-sensitive** due to evolving vLLM CLI
  arguments. The sweep wrapper includes a fail-fast compatibility check to ensure your local vLLM version supports the
  required `--base-url` and JSON parameter grid.

**Installation:** This path requires `vllm` to be installed in your local environment (e.g., `pip install vllm`). The
scripts will call the `vllm` CLI to run the benchmark against the remote endpoint.

Industrial-grade metrics provided by vLLM:

- **TTFT**: Time To First Token
- **TPOT**: Time Per Output Token
- **ITL**: Inter-Token Latency
- **E2EL**: End-to-End Latency
- **Goodput**: Throughput within specific SLOs.

#### What is the Pareto Frontier?

A configuration is on the **Pareto frontier** if you cannot improve one objective (e.g., throughput) without worsening
another (e.g., latency). The frontier shows the best possible tradeoffs for your GPU/model combination.

To use the professional benchmark path:

```bash
# Standard benchmark using vllm bench
BENCHMARK_PATH=vllm BENCHMARK_TIER=latency ./scripts/runpod/demo_end_to_end.sh

# Parameter sweep with Pareto analysis
BENCHMARK_PATH=sweep ./scripts/runpod/demo_end_to_end.sh
```

Results are saved in `results/benchmarks/` and include:

- **vLLM JSON Report**: The raw output from `vllm bench`.
- **Markdown Summary**: A table showing the Pareto frontier (for sweeps).
- **CSV Data**: Full dataset for all tested configurations (for sweeps).

To specify a tier:

```bash
# Via orchestrator
BENCHMARK_TIER=throughput ./scripts/runpod/demo_end_to_end.sh

# Via direct benchmark script
./scripts/benchmark_remote.sh <url> <model> --tier throughput
```

## 6. Run the End-to-End Demo

The orchestrator script supports two modes based on your `.env.runpod` configuration:

### Mode A: Existing Runpod vLLM Pod (Recommended)

Use this if you launched a pod using a **vLLM template** that is already running.

1. Set `RUNPOD_MODE=existing` in `.env.runpod`.
2. Provide `RUNPOD_BASE_URL` (e.g., `https://<pod-id>-8000.proxy.runpod.net/v1`).
3. Set `MODEL_ID` (e.g., `facebook/opt-125m`).
4. Run:
   ```bash
   ./scripts/runpod/demo_end_to_end.sh
   ```

### Mode B: Deploy from Local Machine over SSH (Generic Remote)

Use this to deploy vLLM from your local machine to a **generic GPU VM/pod** (e.g., a PyTorch template).

1. Set `RUNPOD_MODE=generic` in `.env.runpod`.
2. Configure **SSH Strategy** (`RUNPOD_SSH_MODE`):
    - `auto` (Default): Try direct TCP first, then proxied SSH fallback.
    - `direct`: Require direct TCP SSH (SSH over exposed TCP).
    - `proxied`: Use only `ssh.runpod.io` (useful if port 22 is blocked).
3. Provide **SSH Connection Details**:
    - `RUNPOD_SSH_HOST` / `RUNPOD_SSH_PORT`: Direct TCP host and port.
    - `RUNPOD_PROXY_SSH_TARGET`: Proxied SSH target (e.g., `user@ssh.runpod.io`).
4. Provide `RUNPOD_BASE_URL` (e.g., `https://<pod-id>-8000.proxy.runpod.net/v1`).
5. Set `MODEL_ID` (e.g., `facebook/opt-125m`).
6. Run:
   ```bash
   ./scripts/runpod/demo_end_to_end.sh
   ```

> [!TIP]
> **SSH Flexibility**: The orchestrator is designed to work even if direct TCP SSH or SCP/SFTP are unavailable. As long
> as **Proxied SSH** (`ssh.runpod.io`) is configured and working, the script can create remote directories and deploy the
> startup script using **SSH-only file creation** (Base64-encoded transfer).

> [!WARNING]
> **Managed Service Protection**: If the orchestrator detects an already-running vLLM service, it will refuse to mutate
> the pod in `generic` mode by default. You can override this with `RUNPOD_ALLOW_MANAGED_INSTANCE_MUTATION=true` if
> necessary.

The script will:

1. Detect if the host is a managed vLLM service.
2. Verify SSH connectivity (direct or proxied fallback).
3. Deploy the startup script (via SCP or SSH-only fallback).
4. SSH into the pod and start vLLM.
5. Wait for the model to load and the health check to pass via the `RUNPOD_BASE_URL`.
6. Run a benchmark from your local machine against the remote endpoint.

## 7. Manual CLI Operations

If you prefer to run steps individually:

### Start vLLM on the Remote Pod

```bash
# On your local machine
ssh -p <ssh-port> root@<ssh-host> "mkdir -p ~/vlm-inference-lab/scripts/runpod"
scp -P <ssh-port> scripts/runpod/start_vllm.sh root@<ssh-host>:~/vlm-inference-lab/scripts/runpod/
ssh -p <ssh-port> root@<ssh-host> "bash ~/vlm-inference-lab/scripts/runpod/start_vllm.sh facebook/opt-125m"
```

### Run Benchmark Locally

```bash
# Recommended: use the Runpod proxy URL
./scripts/benchmark_remote.sh https://<pod-id>-8000.proxy.runpod.net/v1 facebook/opt-125m
```

## 8. Cleanup

**CRITICAL**: Runpod charges by the hour. When you are finished with the demo:

1. Go to the Runpod Console.
2. **Terminate** the pod (do not just stop it, or you will still be charged for storage).

---

## 💡 Diagnostics & Troubleshooting

Before running the full demo, you can use the lightweight diagnostics tool to check your environment:

```bash
# PowerShell
.\scripts\runpod\vllm_diagnostics.ps1

# Bash
./scripts/runpod/vllm_diagnostics.sh
```

This script reports `RUNPOD_MODE`, checks if `/v1/models` is reachable, verifies SSH/SCP status, and detects if the pod
is already a managed vLLM service.

### Common Issues

- **CUDA Out of Memory**: Choose a smaller model (e.g., `facebook/opt-125m`) or a larger GPU (80GB A100).
- **vLLM Container Fails**: Check the logs on the pod: `docker logs vllm-server`.
