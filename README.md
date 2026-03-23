# VLM Inference Lab: Systems & Optimization

A technical exploration of inference optimization for multimodal (vision-language) models, focusing on GPU performance,
memory behavior, and large-scale serving systems.

## Overview

This repository combines experiments and technical notes around the engineering challenges of deploying modern
Vision-Language Models (VLMs) in production. It bridges the gap between high-level model orchestration and low-level
hardware execution.

> **Key Idea:** Custom kernels matter when framework-level abstractions leave performance on the table, especially for
> memory-bound inference paths and hardware-specific bottlenecks.

**Key Technical Themes:**

- **Inference Systems**:
  [Batching, latency, and throughput](docs/serving/batching_latency_and_throughput.md),
  [KV cache management](docs/serving/kv_cache.md),
  [advanced serving features](docs/serving/advanced_serving_features.md), and
  [training vs inference](docs/serving/training_vs_inference.md).
- **GPU Performance**:
  [GPU memory and kernel execution](docs/systems/gpu_memory_and_kernel_execution.md),
  [profiling and optimization workflow](docs/systems/profiling_and_optimization_workflow.md), and
  [inference kernel bottlenecks](docs/systems/inference_kernel_bottlenecks.md).
- **Transformers and Multimodal Systems**:
  [Transformer tokenization and decoding](docs/transformers/transformers_tokenization_and_decoding.md),
  [position embeddings](docs/transformers/position_embeddings_and_positional_encoding.md),
  [multimodal embeddings](docs/multimodal/embeddings.md),
  [VLM architectures](docs/multimodal/vlm_architectures_and_basics.md), and
  [document understanding](docs/multimodal/document_understanding.md).
- **Hardware Portability**:
  Reasoning about [NVIDIA (CUDA) vs. AMD (HIP/ROCm)](docs/systems/cuda_vs_hip.md) serving.
- **Documentation Backbone**:
  Start with the [documentation map](docs/README.md), which groups the notes by topic and by suggested reading path. For
  the tree-based ML notes, the recommended conceptual order is [decision trees](docs/fundamentals/decision_trees.md) as
  the base learner, [ensemble methods](docs/fundamentals/ensemble_methods.md) as the umbrella theory, then
  [random forests](docs/fundamentals/random_forests.md) and
  [gradient-boosted trees](docs/fundamentals/gradient_boosted_trees.md) as the main tree-ensemble families.

## Repository Layout

```text
vlm-inference-lab/
├── docs/                          # Organized technical notes and reference material
├── source/
│   └── vlm_inference_lab/         # Python orchestration & simulation
├── resources/
│   └── vlm_inference_lab/         # Experimental CUDA/C++ kernels
├── tests/
│   └── vlm_inference_lab/         # Unit tests and regression checks
├── infra/                         # Docker & CI/CD
├── pyproject.toml
└── README.md
```

## Current Experiments

### 1. CUDA Performance: Parallel Reduction

*Location: `resources/vlm_inference_lab/cuda/reduction.cu`*

Demonstrates a structured approach to GPU optimization by moving from a naive atomic-contention implementation to an
optimized shared-memory tree reduction.

- **Measured Result**:
  ```text
  baseline_ms=40.5214
  optimized_ms=0.8101
  speedup=50.019
  ```
- **Optimization Focus**: Memory hierarchy, bank conflicts, and bandwidth vs. synchronization bottlenecks.
- **Read more**: [Profiling and optimization workflow](docs/systems/profiling_and_optimization_workflow.md)

### 2. Serving Simulation: Latency vs. Throughput

*Location: `source/vlm_inference_lab/simulation/inference_simulator.py`*

Models request arrival patterns and dynamic batching policies to analyze P-metrics (P50, P99) under realistic serving
loads.

- **Example Output**:
  ```text
  policy=dynamic_batching
  latency_avg_ms      : 64.2150
  latency_p95_ms      : 142.0210
  throughput_rps      : 88.4200
  ```
- **Key Insight**: How batching timeouts and max batch sizes impact tail latency (P99) in interactive systems.
- **Read more**: [Batching, latency, and throughput](docs/serving/batching_latency_and_throughput.md)

### 3. Cross-Modal Alignment Demo

*Location: `source/vlm_inference_lab/embeddings/demo.py`*

Illustrates the dual-encoder (CLIP-style) architecture using mock embeddings to show how text and image information is
aligned in a shared latent space.

- **Read more**: [Multimodal embeddings](docs/multimodal/embeddings.md)

### 4. Remote Orchestration: Runpod

*Location: `scripts/runpod/demo_end_to_end.ps1` (or `.sh`)*

Supports two modes for remote VLM serving:

- **existing**: Benchmarks a pod already running vLLM via the Runpod proxy URL. No SSH required.
- **generic**: Deploys vLLM from your local machine to a remote GPU host, then benchmarks it.
- **Benchmark Tiers**: Supports `smoke` (10 requests), `latency` (100 requests), `throughput` (200 requests), and
  `sweep` (parameter grid search).
- **Professional Path**: This path uses vLLM's native benchmark CLI. It captures high-fidelity metrics (TTFT, ITL) and
  supports parameter sweeps with Pareto-frontier analysis. The wrappers normalize a trailing `/v1` in the base URL and
  pass explicit endpoints.
- **Diagnostics**: A lightweight helper script (`scripts/runpod/vllm_diagnostics.ps1` / `.sh`) for checking
  connectivity, SSH, and model status.
- **Features**: Automatic model ID resolution, proxied SSH support (`ssh.runpod.io`), SSH-only file creation fallback,
  and managed-instance safety guards.
- **Read more**: [Runpod deployment guide](docs/deployment/runpod_demo.md)

## How to Run

### Python Environment

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install in editable mode
pip install -e .

# Run all tests
pytest
```

### CUDA Benchmarking

To run the automated benchmark (requires `nvcc`):

```powershell
python -m vlm_inference_lab.profiling.cuda_runner --source resources/vlm_inference_lab/cuda/reduction.cu --compile --executable reduction_bench
```

### Inference Simulation

To run the serving simulator:

```powershell
python -m vlm_inference_lab.simulation.inference_simulator
```

### Manual CUDA Execution

To compile and run manually:

```powershell
nvcc resources/vlm_inference_lab/cuda/reduction.cu -o reduction.exe
.
eduction.exe
```

## Status

- ✅ **Structured Benchmarking**: Automated Python-CUDA integration via `CudaRunner`.
- ✅ **System Modeling**: Arrival and batching simulation with robust P-metric reporting.
- ✅ **Technical Docs**: Organized notes on architectures, serving, systems, optimization, and multimodal modeling.
- ✅ **Remote Orchestration**: Production-ready Runpod/vLLM benchmark wrappers.
- ✅ **Reference Docs**:
  - [Documentation map](docs/README.md)
  - [Tree-based ML path: decision trees → ensemble methods → random forests → gradient-boosted trees](docs/fundamentals/ensemble_methods.md)
  - [Neural architecture tradeoffs](docs/architectures/neural_architecture_tradeoffs.md)
  - [RNN gradient stability](docs/architectures/rnn_lstm_gru_and_gradient_stability.md)
  - [VLM architectures](docs/multimodal/vlm_architectures_and_basics.md)
  - [Document understanding](docs/multimodal/document_understanding.md)
  - [Serving optimizations](docs/serving/advanced_serving_features.md)
