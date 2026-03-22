# Training vs. Inference: Systems Perspective

This document explains why the engineering challenges for inference differ significantly from those in training, especially for large-scale multimodal systems.

## Key Differences

| Feature | Training | Inference |
| :--- | :--- | :--- |
| **Primary Goal** | Minimize loss / learn weights | Minimize latency / maximize throughput |
| **Compute Type** | Forward + Backward pass | Forward pass only |
| **Memory usage** | Activations, Gradients, Optimizer states | Weights, KV Cache (for LLMs/VLMs) |
| **Precision** | FP32, BF16, Mixed Precision | FP16, INT8, FP8, 4-bit (Quantization) |
| **Bottleneck** | Often Compute-bound (FLOPs) | Often Memory-bound (Bandwidth) |

## Why Inference is "Harder" for Systems Engineering

1.  **Strict Latency Requirements**: Unlike training, which is an offline process, inference often happens in real-time. If a VLM takes 10 seconds to describe an image, the user experience is poor.
2.  **The KV Cache**: In autoregressive models, we store previous Key-Value pairs to avoid redundant computation. This transforms a compute problem into a **memory capacity and management** problem.
3.  **Dynamic Workloads**: Arrival patterns are unpredictable. Systems must handle "bursty" traffic while maintaining Service Level Objectives (SLOs).
4.  **Hardware Efficiency**: Training uses large, static batches. Inference often uses small, dynamic batches, which makes it harder to saturate the GPU's compute units (CU/SM).

---

## Educational Discussion

> **"Why is transformer inference often memory-bound rather than compute-bound?"**
>
> During the 'generation' phase of a transformer, we produce one token at a time. For each token, we must read the entire model's weights from high-bandwidth memory (HBM) into the GPU caches. Because the amount of computation per weight is small (vector-matrix multiplication), the time spent waiting for memory to arrive at the processor dominates the actual computation time. This is why memory bandwidth is the most critical spec for an inference GPU.
