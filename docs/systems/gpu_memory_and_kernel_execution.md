# GPU Memory and Kernel Execution

This note combines the core CUDA memory hierarchy concepts with the execution model needed to reason about kernel performance on modern GPUs.

## CUDA Memory Model

### Memory Hierarchy

#### Global Memory

The largest, slowest, and most accessible memory (off-chip DRAM).

- Accessible by all threads.
- High latency (hundreds of cycles).
- Requires coalesced access for efficiency.

#### Shared Memory

Small, fast, on-chip memory per-block.

- Low latency (single-digit cycles).
- Used for communication between threads within a block.
- Susceptible to **bank conflicts**.

#### Registers

Fastest on-chip memory.

- Unique to each thread.
- Bottleneck for occupancy (using too many registers per thread reduces the number of active warps).

#### Constant & Texture Memory

Specialized caches for read-only data.

### Memory Bandwidth

The rate at which data can be transferred from global memory to the GPU cores. Many deep learning operations are "
memory-bound" (limited by this bandwidth).

## Shared vs. Global Memory: Optimizing the GPU Memory Hierarchy

Understanding the difference between **Global Memory** (VRAM) and **Shared Memory** (on-chip SRAM) is the key to writing
high-performance kernels.

### Memory Hierarchy Comparison

| Metric        | Global Memory          | Shared Memory                 |
|:--------------|:-----------------------|:------------------------------|
| **Location**  | Off-chip (VRAM)        | On-chip (per-SM)              |
| **Scope**     | Visible to all threads | Visible to threads in a block |
| **Latency**   | High (~400-800 cycles) | Low (~20-30 cycles)           |
| **Bandwidth** | ~1-3 TB/s (H100)       | >10 TB/s                      |
| **Size**      | Large (GBs)            | Tiny (KBs per block)          |

### Tradeoffs: When to use what?

#### Use Shared Memory (SRAM)

- **Data Reuse**: When multiple threads in the same block need to access the same piece of data (e.g., Matrix
  Multiplication, Reduction).
- **Non-Coalesced Access**: When global memory access patterns are scattered, loading a contiguous block into shared
  memory first can "coalesce" the access.
- **Tiling**: Breaking a large problem into small "tiles" that fit in shared memory.

#### Use Global Memory (VRAM)

- **Streaming Data**: For simple operations where each element is read only once (e.g., Vector Add).
- **Large Datasets**: When the required data exceeds the small capacity of shared memory (~64-100KB per block).

### Common Performance Pitfalls

- **Shared Memory Bank Conflicts**: Simultaneous access to different addresses in the same memory bank by multiple
  threads in a warp. This causes the access to be serialized.
- **Global Memory Coalescing**: Ensuring consecutive threads access consecutive memory addresses to maximize bandwidth
  utilization.

---

### Educational Discussion

> **"Why does shared memory matter for performance?"**
>
> Shared memory is on-chip, meaning it has much lower latency and much higher bandwidth than global VRAM. If a kernel is
> memory-bound (like most inference operators), we want to minimize global memory reads.
>
> In our **Reduction benchmark**, for example, we load data into shared memory once and then do the reduction in-place.
> This reduces the number of global memory writes from one per thread to one per block. This 'tiling' strategy is the
> foundation of almost all optimized kernels in libraries like cuBLAS or FlashAttention.

## Kernel Execution: Mapping CUDA to Hardware

A **Kernel** is the basic unit of work for a GPU, representing a single function that is executed in parallel across
many threads.

### SIMT (Single Instruction, Multiple Threads)

The hardware manages threads in groups of 32, called **warps**. All threads in a warp execute the same instruction
simultaneously.

### Thread Hierarchy & Mapping

1. **Grid**: The collection of all thread blocks in a kernel launch.
2. **Block**: A group of threads that can share data (via shared memory) and synchronize (via `__syncthreads()`). Blocks
   are scheduled on **Streaming Multiprocessors (SMs)**.
3. **Thread**: The smallest unit of execution.

**Launch Configuration**: `kernel<<<blocksPerGrid, threadsPerBlock>>>(args)`

### Performance Concepts

- **Occupancy**: The ratio of active warps to the maximum possible active warps per SM. High occupancy is the primary
  way GPUs **hide memory latency**.
- **Warp Divergence**: Occurs when threads in the same warp take different branches (e.g., `if-else`). The warp must
  execute all paths sequentially, "masking" inactive threads, which wastes compute.
- **Memory Coalescing**: When consecutive threads in a warp access consecutive memory addresses, the hardware "
  coalesces" these into a single memory transaction. This is the most critical optimization for memory-bound kernels.

---

### Educational Discussion

> **"What is a GPU kernel, and how does the hardware execute it?"**
>
> A kernel is a function written in CUDA that runs in parallel. When you launch a kernel, you specify a grid of blocks,
> and each block contains many threads. The GPU schedules these blocks across its Streaming Multiprocessors (SMs).
>
> Inside an SM, threads are grouped into **warps of 32**. This is the fundamental unit of scheduling. The most important
> thing to remember is that threads in a warp execute the same instruction. If your code has lots of branches
> (divergence) or random memory accesses (non-coalesced), you'll lose performance because the hardware can't efficiently
> group the work.
