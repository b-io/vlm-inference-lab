# Batching Strategies for Modern Inference

Efficient serving of Vision-Language Models (VLMs) requires clever batching to saturate GPU resources without causing
excessive latency.

## Static vs. Dynamic Batching

### Static Batching (Naive)

Fixed batch size used during inference.

- **Problem**: If the arrival rate is low, the GPU sits idle. If the arrival rate is high, queues form. Inefficient for
  real-world serving.

### Dynamic Batching (Server-side)

Combines multiple incoming requests into a single batch based on a **timeout** and **max batch size**.

- **Balanced**: Improves GPU utilization while keeping latency within SLO limits.

## The State-of-the-Art: Continuous Batching

Standard in modern serving frameworks like **vLLM** and **SGLang**.

- **Problem**: Requests have different sequence lengths. In traditional batching, short requests wait for the longest
  one to finish.
- **Solution**: Tokens are generated in a continuous stream. New requests can join the batch at each token step, and
  finished requests can leave immediately.
- **Impact**: Up to **10x higher throughput** than traditional static batching by eliminating idle GPU cores.

---

## Educational Discussion

> **"What is the difference between static and continuous batching?"**
>
> Static batching requires all requests to start and finish at the same time. This is very inefficient for VLMs because
> different prompts lead to different generation lengths. Short requests get "trapped" waiting for the longest request
> in the batch to finish.
>
> **Continuous batching** (also called Iteration-level batching) solves this by scheduling at the granularity of a
> single token iteration. This means a new request can be added to the running batch as soon as any other request
> finishes. This dramatically reduces idle time and allows us to serve many more users on the same hardware.
