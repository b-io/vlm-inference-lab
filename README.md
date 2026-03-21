# VLM Inference Lab: Systems & Optimization

A technical exploration of inference optimization for multimodal (vision-language) models, focusing on GPU performance, memory behavior, and large-scale serving systems.

## Overview

This repository is a collection of experiments and technical notes around the engineering challenges of deploying modern Vision-Language Models (VLMs) in production. It bridges the gap between high-level model orchestration and low-level hardware execution.

> **Key Idea:** Custom kernels matter when framework-level abstractions leave performance on the table, especially for memory-bound inference paths and hardware-specific bottlenecks.

**Key Technical Themes:**
- **Inference Systems**: Batching policies, KV cache management, and latency-throughput tradeoffs.
- **GPU Performance**: Custom CUDA kernels, memory-bound vs. compute-bound analysis.
- **Multimodal Alignment**: Shared embedding spaces and cross-modal reasoning architectures.
- **Hardware Portability**: Reasoning about NVIDIA (CUDA) vs. AMD (HIP/ROCm) serving.

## Repository Layout

```text
vlm-inference-lab/
├── docs/                          # Technical notes & Interview discussion points
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

Demonstrates a structured approach to GPU optimization by moving from a naive atomic-contention implementation to an optimized shared-memory tree reduction.
- **Measured Result**:
  ```text
  baseline_ms=40.5214
  optimized_ms=0.8101
  speedup=50.019
  ```
- **Interview Focus**: Memory hierarchy, bank conflicts, and bandwidth vs. synchronization bottlenecks.
- **Read more**: [Baseline vs. Optimized Analysis](docs/profiling/baseline_vs_optimized.md)

### 2. Serving Simulation: Latency vs. Throughput
*Location: `source/vlm_inference_lab/simulation/inference_simulator.py`*

Models request arrival patterns and dynamic batching policies to analyze P-metrics (P50, P99) under realistic serving loads.
- **Example Output**:
  ```text
  policy=dynamic_batching
  latency_avg_ms      : 64.2150
  latency_p95_ms      : 142.0210
  throughput_rps      : 88.4200
  ```
- **Key Insight**: How batching timeouts and max batch sizes impact tail latency (P99) in interactive systems.
- **Read more**: [Latency vs. Throughput Tradeoffs](docs/inference/latency_vs_throughput.md)

### 3. Cross-Modal Alignment Demo
*Location: `source/vlm_inference_lab/embeddings/demo.py`*

Illustrates the 'Dual Encoder' (CLIP-style) architecture using mock embeddings to show how text and image information is aligned in a shared latent space.
- **Read more**: [Multimodal Embeddings](docs/vlm/embeddings.md)

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
.\reduction.exe
```

## Status
✅ **Structured Benchmarking**: Automated Python-CUDA integration via `CudaRunner`.
✅ **System Modeling**: Arrival/Batching simulation with robust P-metric reporting.
✅ **Technical Docs**: Concise notes on KV cache, batching, and hardware-aware optimization.
🚧 **Real Profiling**: (In Progress) Integrating NSight CLI for automated metric capture.
