# Batching, Latency, and Throughput

This note combines batching strategies with the latency-throughput tradeoff they create in production inference systems.

## Batching Strategies for Modern Inference

Efficient serving of Vision-Language Models (VLMs) requires clever batching to saturate GPU resources without causing
excessive latency.

### Static vs. Dynamic Batching

#### Static Batching (Naive)

Fixed batch size used during inference.

- **Problem**: If the arrival rate is low, the GPU sits idle. If the arrival rate is high, queues form. Inefficient for
  real-world serving.

#### Dynamic Batching (Server-side)

Combines multiple incoming requests into a single batch based on a **timeout** and **max batch size**.

- **Balanced**: Improves GPU utilization while keeping latency within SLO limits.

### The State-of-the-Art: Continuous Batching

Standard in modern serving frameworks like **vLLM** and **SGLang**.

- **Problem**: Requests have different sequence lengths. In traditional batching, short requests wait for the longest
  one to finish.
- **Solution**: Tokens are generated in a continuous stream. New requests can join the batch at each token step, and
  finished requests can leave immediately.
- **Impact**: Up to **10x higher throughput** than traditional static batching by eliminating idle GPU cores.

---

### Educational Discussion

> **"What is the difference between static and continuous batching?"**
>
> Static batching requires all requests to start and finish at the same time. This is very inefficient for VLMs because
> different prompts lead to different generation lengths. Short requests get "trapped" waiting for the longest request
> in the batch to finish.
>
> **Continuous batching** (also called Iteration-level batching) solves this by scheduling at the granularity of a
> single token iteration. This means a new request can be added to the running batch as soon as any other request
> finishes. This dramatically reduces idle time and allows us to serve many more users on the same hardware.

## Latency vs Throughput in Inference Systems

This document explores the fundamental tradeoff between individual request latency and overall system throughput in
multimodal inference serving.

### The Core Tradeoff: Batching

In Vision-Language Models (VLMs), GPU utilization is highly dependent on batch size. Larger batches typically increase
**throughput** (more requests per second) but also increase **latency** (each request waits longer for the batch to fill
and takes longer to process).

#### Simulation Results (Reference Run)

Using `source/vlm_inference_lab/simulation/inference_simulator.py`, we can model different serving scenarios:

| Scenario            | Arrival Rate | Max Batch | Timeout | P50 Latency | P99 Latency | Throughput |
|:--------------------|:-------------|:----------|:--------|:------------|:------------|:-----------|
| **Low Latency**     | 5.0 rps      | 1         | 0ms     | 18.25 ms    | 24.10 ms    | 4.95 rps   |
| **Balanced**        | 20.0 rps     | 8         | 50ms    | 45.30 ms    | 98.45 ms    | 19.82 rps  |
| **High Throughput** | 60.0 rps     | 32        | 150ms   | 185.10 ms   | 310.20 ms   | 59.10 rps  |

*Note: Results obtained with `seed=42` and `gpu_throughput_per_unit=2000`.*

#### Key Observations

1. **Queue Time vs Service Time**: As the arrival rate approaches the maximum system capacity, queue time (the time a
   request spends waiting for a batcher or a free GPU) grows exponentially.
2. **The "Tail"**: P99 latency is often much higher than P50 due to batching timeouts and request bursts.
3. **Efficiency**: Large batches are more efficient for the GPU (better memory coalescing and computation-to-overhead
   ratio), but they are only useful if there is enough traffic to fill them without hitting long timeouts.
4. **Prefill vs Decode**: Prefill costs scale linearly with request count, while decode costs scale with token count.
   The simulator models this to show how sequence length impacts serving efficiency.

### Evaluation Metrics

- **Throughput (rps)**: Total requests processed per second.
- **Latency P50 (ms)**: Median response time.
- **Latency P99 (ms)**: Tail latency (99th percentile).
- **Service Time**: Actual time spent on GPU.
- **Queue Time**: Time spent waiting to be batched or processed.

### Practical discussion

> **Reasoning about latency vs throughput**
>
> In a production inference system, we can't optimize for both simultaneously. If we want the lowest possible latency,
> we process requests as they arrive (Batch Size 1), but this is very expensive and inefficient for the GPU. If we want
> the highest throughput, we wait for large batches, which penalizes the first request in the batch.
>
> In real systems (like vLLM), we use **Continuous Batching** and **Dynamic Batching** to find a middle ground—adjusting
> batching policies based on current traffic and hardware constraints. I use simulators to model these P-metrics (P50,
> P99) to ensure that we meet our Service Level Objectives (SLOs) while maintaining high hardware utilization.
