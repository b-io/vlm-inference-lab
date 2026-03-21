# Profiling Foundations: Measure Before You Optimize

In performance engineering, the first rule is **Measure First**. Profiling is the process of identifying where time and memory are being spent in a system.

## The Hierarchy of Profiling Tools

1.  **System-Level (NSight Systems / `nsys`)**:
    - **Focus**: The whole system timeline.
    - **Questions**: "Is the GPU idle?", "Are we waiting for data from the CPU (H2D)?", "Where are the gaps between kernels?"
2.  **Kernel-Level (NSight Compute / `ncu`)**:
    - **Focus**: A single GPU kernel's execution.
    - **Questions**: "Is this kernel compute-bound or memory-bound?", "What is our SM occupancy?", "Are there bank conflicts in shared memory?"
3.  **Application-Level (PyTorch Profiler / `torch.profiler`)**:
    - **Focus**: Python-to-CUDA orchestration.
    - **Questions**: "Which PyTorch operator is the bottleneck?", "How much time is spent in CUDA graph overhead?"

## Common Bottleneck Signals

- **High Latency, Low Memory Throughput**: Likely a **compute-bound** kernel (e.g., complex math) or high **synchronization overhead** (e.g., too many `__syncthreads()`).
- **Low Latency, High Memory Throughput**: Likely a **memory-bound** kernel (e.g., Vector Add, LayerNorm). Performance is limited by how fast we can pull data from HBM.
- **Large Gaps in Timeline**: Likely a **CPU bottleneck**. The GPU is fast, but the CPU can't schedule kernels quickly enough.

---

## Interview Discussion

> **"How do you approach profiling a slow inference pipeline?"**
>
> I start top-down. First, I use a tool like **NSight Systems** to see the overall timeline. I'm looking for 'gaps' where the GPU is idle, which signals a CPU bottleneck or high H2D copy time.
>
> If the GPU is busy but slow, I'll use **NSight Compute** to analyze the most expensive kernels. I'll check the **roofline model** to see if we're hitting a memory bandwidth ceiling or a compute ceiling. For most VLM operators (like softmax or layernorm), the bottleneck is usually memory bandwidth, which is why I focus on optimizations like **operator fusion** to reduce redundant memory reads.
