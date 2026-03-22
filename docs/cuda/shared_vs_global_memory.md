# Shared vs. Global Memory: Optimizing the GPU Memory Hierarchy

Understanding the difference between **Global Memory** (VRAM) and **Shared Memory** (on-chip SRAM) is the key to writing
high-performance kernels.

## Memory Hierarchy Comparison

| Metric        | Global Memory          | Shared Memory                 |
|:--------------|:-----------------------|:------------------------------|
| **Location**  | Off-chip (VRAM)        | On-chip (per-SM)              |
| **Scope**     | Visible to all threads | Visible to threads in a block |
| **Latency**   | High (~400-800 cycles) | Low (~20-30 cycles)           |
| **Bandwidth** | ~1-3 TB/s (H100)       | >10 TB/s                      |
| **Size**      | Large (GBs)            | Tiny (KBs per block)          |

## Tradeoffs: When to use what?

### Use Shared Memory (SRAM)

- **Data Reuse**: When multiple threads in the same block need to access the same piece of data (e.g., Matrix
  Multiplication, Reduction).
- **Non-Coalesced Access**: When global memory access patterns are scattered, loading a contiguous block into shared
  memory first can "coalesce" the access.
- **Tiling**: Breaking a large problem into small "tiles" that fit in shared memory.

### Use Global Memory (VRAM)

- **Streaming Data**: For simple operations where each element is read only once (e.g., Vector Add).
- **Large Datasets**: When the required data exceeds the small capacity of shared memory (~64-100KB per block).

## Common Performance Pitfalls

- **Shared Memory Bank Conflicts**: Simultaneous access to different addresses in the same memory bank by multiple
  threads in a warp. This causes the access to be serialized.
- **Global Memory Coalescing**: Ensuring consecutive threads access consecutive memory addresses to maximize bandwidth
  utilization.

---

## Educational Discussion

> **"Why does shared memory matter for performance?"**
>
> Shared memory is on-chip, meaning it has much lower latency and much higher bandwidth than global VRAM. If a kernel is
> memory-bound (like most inference operators), we want to minimize global memory reads.
>
> In our **Reduction benchmark**, for example, we load data into shared memory once and then do the reduction in-place.
> This reduces the number of global memory writes from one per thread to one per block. This 'tiling' strategy is the
> foundation of almost all optimized kernels in libraries like cuBLAS or FlashAttention.
