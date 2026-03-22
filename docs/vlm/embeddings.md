# Multimodal Embeddings & Cross-Modal Alignment

This document explains how Vision-Language Models (VLMs) bridge the gap between visual and textual information using
shared embedding spaces.

## Concept: The Shared Latent Space

In a multimodal system, we want to represent an image and its corresponding description as vectors that are "close" to
each other in a high-dimensional space.

### 1. Dual-Encoder Models (e.g., CLIP)

- **Architecture**: Separate encoders for image (e.g., ViT) and text (e.g., Transformer).
- **Mechanism**: Both outputs are projected into a shared space of dimension $D$.
- **Inference**: High-speed retrieval using **cosine similarity**.
- **Pros**: Extremely fast for searching millions of images; allows for zero-shot classification.
- **Cons**: Limited "reasoning" capability; the model only knows if an image matches a text, not *why*.

### 2. Fusion / Generative Models (e.g., LLaVA, Flamingo)

- **Architecture**: A vision encoder connected to a Large Language Model (LLM) via a **projection layer** or
  **adapter**.
- **Mechanism**: Visual features are treated as "visual tokens" and concatenated with text tokens.
- **Inference**: The LLM "reads" the image tokens to generate a response.
- **Pros**: Deep reasoning, complex VQA, and document understanding.
- **Cons**: Much higher inference latency than dual-encoders.

## Alignment & The Modality Gap

Even after training, image and text embeddings often cluster in different regions of the latent space—this is known as
the **Modality Gap**.

**Techniques to improve alignment**:

- **Contrastive Learning**: Training on millions of (image, caption) pairs.
- **Projection Layers**: Learning a linear or MLP mapping to align vision features with the LLM's word embedding space.
- **Fine-tuning**: Using high-quality instructions (e.g., "Describe this image in detail") to teach the model how to
  ground text in visual features.

---

## Educational Discussion

> **"How do image and text embeddings relate in a VLM?"**
>
> It depends on the architecture. In retrieval-style models like CLIP, we use a dual-encoder setup where both modalities
> are projected into a shared space, and we measure alignment using cosine similarity. This is great for scale and
> speed.
>
> However, for modern generative VLMs like LLaVA, we use a projection layer to map visual features into the LLM's token
> space. Here, 'alignment' means the LLM can interpret visual tokens as if they were words. This enables complex
> reasoning, but the inference cost is significantly higher because we're running a full autoregressive LLM.
