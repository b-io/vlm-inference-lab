# Vision-Language Models (VLM): Architecture & Systems

Vision-Language Models (VLMs) combine visual and textual understanding, typically by connecting a **Vision Encoder** with a **Large Language Model (LLM)**.

## Architecture Comparison

| Model Pattern | Typical Use Case | Strength | Serving Implication |
| :--- | :--- | :--- | :--- |
| **Dual-Encoder** (e.g., CLIP) | Image-text retrieval, zero-shot classification | Extremely fast similarity search | Compute-bound for large batches, but low memory pressure |
| **Fusion / Generative** (e.g., LLaVA) | VQA, visual grounding, document reasoning | High reasoning capability | Memory-bound by KV cache, high 'visual token inflation' |

## Core Architecture Patterns

1.  **Dual-Encoder (e.g., CLIP)**: Two separate encoders projected into a shared space. Ideal for **retrieval** and **zero-shot classification**.
2.  **Generative / Fusion (e.g., LLaVA, Llama-Vision)**: A vision encoder (like ViT) is connected to an LLM via a **projector** (e.g., MLP or Q-Former). The LLM processes "visual tokens" as if they were text.

## Systems Challenges for VLMs

### Image Resolution vs. Memory
Higher resolution images (e.g., 1024x1024) require more "visual tokens."
- **Example**: If an image is broken into 14x14 patches, it produces 196 tokens. If each token is a vector of 1024 floats, an image adds significant memory pressure to the KV cache before generation even begins.

### Document Understanding & VQA
Tasks like **Document OCR** and **Visual Question Answering (VQA)** require the model to ground its text generation in specific pixel regions.
- **Problem**: This requires fine-grained spatial information, which often means more tokens and higher inference latency.

---

## Interview Discussion

> **"What are the main components of a VLM, and why are they hard to serve?"**
>
> A generative VLM has three parts: a vision encoder (like ViT), an LLM (like Llama), and a projector (like an MLP) that aligns them.
>
> They are hard to serve because of the 'visual token inflation.' A single image can easily become 576 or more tokens. Before the LLM even generates its first word, the KV cache is already partially filled. This limits the **batch size** we can use and makes the **TTFT (Time To First Token)** much higher than a text-only model. To optimize this, we use techniques like **KV cache sharing** and **low-precision quantization (INT8/FP8)** for both the model weights and the KV cache.
