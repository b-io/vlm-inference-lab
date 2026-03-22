# KV Cache: The Memory Constraint of Modern Inference

In multimodal and language models, the **KV Cache** is the single most important factor for inference memory management.

## Why it Exists

During autoregressive generation (one token at a time), each new token needs the attention keys (K) and values (V) of
all *previous* tokens. To avoid re-computing these for every step ($O(N^2)$ computation), we store them in a
cache ($O(N)$ computation).

## The Memory Problem

The KV cache grows linearly with **Batch Size** and **Sequence Length**. For long-form document understanding or
high-resolution visual reasoning, the KV cache can easily exceed the size of the model weights.

**Memory Footprint Calculation**:
$2 \times \text{layers} \times \text{hidden size} \times \text{seq length} \times \text{batch size} \times \text{precision (bytes)}$

## Architecture Optimization

- **Multi-Query Attention (MQA)**: One K/V head for all query heads. Great for memory, but can hurt accuracy.
- **Grouped-Query Attention (GQA)**: One K/V head per *group* of query heads. Used in **Llama 3**.
- **PagedAttention**: Managing the KV cache in non-contiguous blocks (like OS paging) to eliminate internal
  fragmentation. This is the core innovation behind **vLLM**.

---

## Educational Discussion

> **"Why does the KV cache create memory pressure, and how do we solve it?"**
>
> The KV cache grows linearly with the sequence length. If you're processing a long document (e.g., 32k tokens), the KV
> cache for a single request can consume gigabytes of VRAM. This limits the number of concurrent requests we can serve (
> throughput).
>
> To solve this, we use techniques like **PagedAttention** to minimize fragmentation and allow for dynamic memory
> allocation. We also see architectural shifts like **Grouped-Query Attention (GQA)**, which reduces the KV cache size by
> a factor of 8 or more by sharing Key and Value heads across multiple Query heads. This directly improves the throughput
> we can achieve on a single GPU.
