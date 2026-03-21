# CUDA Memory Model

## Memory Hierarchy

### Global Memory
The largest, slowest, and most accessible memory (off-chip DRAM).
- Accessible by all threads.
- High latency (hundreds of cycles).
- Requires coalesced access for efficiency.

### Shared Memory
Small, fast, on-chip memory per-block.
- Low latency (single-digit cycles).
- Used for communication between threads within a block.
- Susceptible to **bank conflicts**.

### Registers
Fastest on-chip memory.
- Unique to each thread.
- Bottleneck for occupancy (using too many registers per thread reduces the number of active warps).

### Constant & Texture Memory
Specialized caches for read-only data.

## Memory Bandwidth
The rate at which data can be transferred from global memory to the GPU cores. Many deep learning operations are "memory-bound" (limited by this bandwidth).
