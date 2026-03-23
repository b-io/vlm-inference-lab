# Profiling and Optimization Workflow

This note combines profiling foundations with a concrete optimization case study so the measurement workflow and the
resulting performance gains can be understood together.

## Profiling Foundations: Measure Before You Optimize

In performance engineering, the first rule is **Measure First**. Profiling is the process of identifying where time and
memory are being spent in a system.

### The Hierarchy of Profiling Tools

1. **System-Level (NSight Systems / `nsys`)**:
    - **Focus**: The whole system timeline.
    - **Questions**: "Is the GPU idle?", "Are we waiting for data from the CPU (H2D)?", "Where are the gaps between
      kernels?"
2. **Kernel-Level (NSight Compute / `ncu`)**:
    - **Focus**: A single GPU kernel's execution.
    - **Questions**: "Is this kernel compute-bound or memory-bound?", "What is our SM occupancy?", "Are there bank
      conflicts in shared memory?"
3. **Application-Level (PyTorch Profiler / `torch.profiler`)**:
    - **Focus**: Python-to-CUDA orchestration.
    - **Questions**: "Which PyTorch operator is the bottleneck?", "How much time is spent in CUDA graph overhead?"

### Common Bottleneck Signals

- **High Latency, Low Memory Throughput**: Likely a **compute-bound** kernel (e.g., complex math) or high
  **synchronization overhead** (e.g., too many `__syncthreads()`).
- **Low Latency, High Memory Throughput**: Likely a **memory-bound** kernel (e.g., Vector Add, LayerNorm). Performance
  is limited by how fast we can pull data from HBM.
- **Large Gaps in Timeline**: Likely a **CPU bottleneck**. The GPU is fast, but the CPU can't schedule kernels quickly
  enough.

---

### Educational Discussion

> **"How do you approach profiling a slow inference pipeline?"**
>
> I start top-down. First, I use a tool like **NSight Systems** to see the overall timeline. I'm looking for 'gaps'
> where the GPU is idle, which signals a CPU bottleneck or high H2D copy time.
>
> If the GPU is busy but slow, I'll use **NSight Compute** to analyze the most expensive kernels. I'll check the
> **roofline model** to see if we're hitting a memory bandwidth ceiling or a compute ceiling. For most VLM operators
> (like softmax or layernorm), the bottleneck is usually memory bandwidth, which is why I focus on optimizations like
> **operator fusion** to reduce redundant memory reads.

## Baseline vs Optimized Performance

This document records the methodology and results for the CUDA parallel reduction benchmark, demonstrating a structured
approach to performance engineering.

### Benchmarking Philosophy

Always establish a baseline before applying any "optimization". Measure twice, cut once. Use structured runners (like
`cuda_runner.py`) to ensure reproducibility.

### Case Study: Parallel Reduction (Sum)

*Experiment located at: `resources/vlm_inference_lab/cuda/reduction.cu`*

Reduction is a classic example of a communication-bound operation. The performance gap between naive and optimized
versions is often several orders of magnitude.

#### 1. Naive Implementation

- **Method**: Each thread uses `atomicAdd` on a single global memory location.
- **Problem**: Extreme contention. Thousands of threads trying to update one memory address causes serialized execution
  at the memory controller level.
- **Complexity**: O(N) global memory atomic operations.

#### 2. Optimized Implementation (Shared Memory & Tree Reduction)

- **Method**: Shared memory tiling and tree-based reduction within blocks.
- **Improvements**:
    - **Coalesced global memory loads**: Threads read consecutive memory addresses.
    - **Shared memory tiling**: Uses fast on-chip memory for intermediate sums.
    - **Tree-based reduction**: Reduces O(N) to O(log N) operations within each block.
    - **Minimized global memory writes**: Only 1 atomic add per block (e.g., 256x reduction in atomic operations).

#### 3. Measured Results (Reference Run)

*Environment: NVIDIA GeForce RTX 4090 | Windows 11 | CUDA 12.4*

```text
Benchmarking Reduction | N = 16777216 | Threads/Block = 256

--- Performance Results ---
Naive:      40.5214 ms | Sum: 16777216.000000 (OK)
Optimized:   0.8101 ms | Sum: 16777216.000000 (OK)
Speedup:      50.02x

METRICS_START
baseline_ms=40.5214
optimized_ms=0.8101
speedup=50.019
correct=true
METRICS_END
```

| Implementation         | Execution Time (ms) | Speedup | Notes                                                      |
|:-----------------------|:--------------------|:--------|:-----------------------------------------------------------|
| **Naive (Atomic)**     | 40.5214 ms          | 1.00x   | Bottleneck: Extreme atomic contention on a single address. |
| **Optimized (Shared)** | 0.8101 ms           | 50.02x  | Bottleneck: Global memory bandwidth (ideal).               |

**Execution Command:**

```powershell
python -m vlm_inference_lab.profiling.cuda_runner --source resources/vlm_inference_lab/cuda/reduction.cu --compile --executable reduction_bench
```

#### 4. Interpretation & Next Steps

- **Why the naive version is slower**: Thousands of threads (65,536 blocks * 256 threads) all attempting to `atomicAdd`
  to a *single* memory address in global memory. This causes massive serialization at the memory controller level.
- **What changed in the optimized version**: Each thread first loads its data into **Shared Memory**. Then, a
  **Tree-based reduction** is performed *on-chip*. Finally, only **one** `atomicAdd` per block is performed to global
  memory. This reduces global atomic contention by 256x.
- **Bottleneck remaining**: At ~0.8ms for 16.7M elements (67MB), we are hitting ~83 GB/s. On high-end cards, the
  bottleneck is now purely the **Global Memory Bandwidth**.
- **Next Optimization**: To further improve, one could use **Shuffle instructions (`__shfl_down_sync`)** to avoid shared
  memory bank conflicts and further reduce latency within a warp.

### Practical discussion

> **How to optimize a GPU workload**
>
> In this experiment, I identified that the bottleneck was atomic contention. By switching to a tree-based reduction in
> shared memory, I reduced the number of global memory atomic operations by a factor of 256. This shifted the bottleneck
> from synchronization overhead to memory bandwidth, which is the ideal state for a communication-heavy reduction
> operation. It demonstrates the importance of understanding the memory hierarchy—specifically moving high-frequency
> operations from global memory to shared memory.
> ... (rest of the file)
