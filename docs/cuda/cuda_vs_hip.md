# CUDA vs HIP: Portability and Performance

This document explains the practical differences between NVIDIA and AMD hardware for multimodal inference and how to reason about portability.

## Overview
As the AI hardware landscape expands beyond NVIDIA, understanding **HIP (Heterogeneous-compute Interface for Portability)** is crucial. HIP is AMD's C++ runtime API that allows developers to write portable code that can run on both NVIDIA and AMD GPUs.

## Key Differences

| Feature | NVIDIA (CUDA) | AMD (ROCm/HIP) |
| :--- | :--- | :--- |
| **Compiler** | `nvcc` | `hipcc` |
| **Hardware** | H100, A100, RTX 4090 | MI300X, MI250, RX 7900 XTX |
| **API Prefix** | `cuda*` (e.g., `cudaMalloc`) | `hip*` (e.g., `hipMalloc`) |
| **Kernel Launch** | `kernel<<<...>>>` | `hipLaunchKernelGGL(...)` or `<<<...>>>` |

## Practical Portability with `hipify`
AMD provides tools like `hipify-clang` that automatically converts CUDA source code into HIP source code. For many kernels, this is a 1:1 mapping of function names (e.g., `cudaMalloc` -> `hipMalloc`).

## H100 vs MI300X: Inference Perspective
When deploying Vision-Language Models (VLMs), the hardware choice impacts the serving strategy:

- **Memory Capacity**: The AMD MI300X offers **192GB HBM3**, while the NVIDIA H100 typically has **80GB HBM3**. This is a massive difference for VLM inference, where the **KV Cache** for long-document understanding or high-resolution images can quickly consume VRAM.
- **Memory Bandwidth**: Both offer extremely high bandwidth (~3-5 TB/s), which is the primary bottleneck for autoregressive decoding (the "generation" phase of a VLM).
- **Ecosystem**: NVIDIA's TensorRT-LLM is highly optimized for H100, while ROCm is the primary stack for MI300X, supported by frameworks like vLLM and SGLang.

## Educational Discussion

> **"How would you think about portability across H100 and MI300X?"**
>
> From a software engineering perspective, we want to avoid hardware lock-in. For high-level orchestration, we rely on frameworks like PyTorch or Triton that abstract the backend. For custom kernels, we can use HIP to write code that runs on both.
>
> From a systems perspective, the MI300X's larger memory capacity (192GB) is very attractive for VLMs because it allows for much larger batch sizes or longer context windows (KV cache) on a single node compared to the H100. However, the decision also depends on the maturity of the optimization stack (like TensorRT-LLM vs. ROCm) and the specific model's compute-to-memory ratio.
