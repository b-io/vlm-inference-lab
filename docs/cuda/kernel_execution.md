# Kernel Execution: Mapping CUDA to Hardware

A **Kernel** is the basic unit of work for a GPU, representing a single function that is executed in parallel across many threads.

## SIMT (Single Instruction, Multiple Threads)
The hardware manages threads in groups of 32, called **warps**. All threads in a warp execute the same instruction simultaneously.

## Thread Hierarchy & Mapping

1.  **Grid**: The collection of all thread blocks in a kernel launch.
2.  **Block**: A group of threads that can share data (via shared memory) and synchronize (via `__syncthreads()`). Blocks are scheduled on **Streaming Multiprocessors (SMs)**.
3.  **Thread**: The smallest unit of execution.

**Launch Configuration**: `kernel<<<blocksPerGrid, threadsPerBlock>>>(args)`

## Performance Concepts

- **Occupancy**: The ratio of active warps to the maximum possible active warps per SM. High occupancy is the primary way GPUs **hide memory latency**.
- **Warp Divergence**: Occurs when threads in the same warp take different branches (e.g., `if-else`). The warp must execute all paths sequentially, "masking" inactive threads, which wastes compute.
- **Memory Coalescing**: When consecutive threads in a warp access consecutive memory addresses, the hardware "coalesces" these into a single memory transaction. This is the most critical optimization for memory-bound kernels.

---

## Interview Discussion

> **"What is a GPU kernel, and how does the hardware execute it?"**
>
> A kernel is a function written in CUDA that runs in parallel. When you launch a kernel, you specify a grid of blocks, and each block contains many threads. The GPU schedules these blocks across its Streaming Multiprocessors (SMs).
>
> Inside an SM, threads are grouped into **warps of 32**. This is the fundamental unit of scheduling. The most important thing to remember is that threads in a warp execute the same instruction. If your code has lots of branches (divergence) or random memory accesses (non-coalesced), you'll lose performance because the hardware can't efficiently group the work.
