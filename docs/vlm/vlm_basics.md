# Vision-Language Models (VLM): Architecture & Systems

Vision-Language Models (VLMs) combine **visual understanding** with **language understanding and generation**. A VLM takes visual input such as an image, screenshot, chart, or document page and uses that information to retrieve, classify, ground, explain, or generate text.

A useful mental model is:

- a **vision module** that turns images into visual features or visual tokens
- a **language module** that reasons over text and often generates output
- an **alignment or fusion mechanism** that lets the model connect the two modalities

VLMs are not all the same. Some are optimized for **retrieval**, others for **grounded reasoning**, and others for **multimodal generation**. Those choices strongly affect both model behavior and serving cost.

## What problems VLMs solve

Typical VLM tasks include:

- image-text retrieval
- zero-shot image classification
- image captioning
- visual question answering (VQA)
- multimodal chat
- document understanding
- screenshot and UI understanding
- grounded generation over images, pages, or diagrams

## Architecture taxonomy

### 1. Dual-encoder / contrastive models

**Examples:** CLIP, SigLIP

These models use one encoder for the image and one encoder for the text, then map both into a shared embedding space.

**Best for**
- image-text retrieval
- zero-shot classification
- embedding search
- reranking

**Strengths**
- simple and scalable
- very fast at inference for retrieval use cases
- easy to precompute embeddings and index them offline

**Weaknesses**
- not naturally generative
- weaker token-level grounding and reasoning than fused or generative models
- not ideal for multimodal assistants

**Serving implication**
- usually lower KV-cache pressure than generative VLMs
- often compute-bound for large embedding batches
- a strong choice when the product is search or ranking rather than chat

### 2. Fusion / cross-attention models

**Examples:** ALBEF-style systems, Flamingo-style cross-attention bridges

These models allow image features and text features to interact through cross-attention or multimodal fusion blocks.

**Best for**
- visual question answering
- grounding
- interleaved image-text conditioning
- few-shot multimodal prompting

**Strengths**
- stronger visual grounding than dual encoders
- better when the text must attend to specific objects or regions

**Weaknesses**
- more complex and more expensive than dual encoders
- less naturally optimized for large-scale retrieval systems

**Serving implication**
- more expensive than contrastive models
- often a middle ground between retrieval models and full multimodal assistants

### 3. Vision encoder + projector + LLM

**Examples:** LLaVA and many modern multimodal assistants

This is the most common modern pattern:

- a vision encoder extracts visual tokens
- a projector or adapter maps them into the LLM embedding space
- a decoder-only LLM consumes both text tokens and visual tokens and generates text

**Best for**
- multimodal chat
- captioning
- VQA
- assistant-style interfaces

**Strengths**
- reuses strong pretrained LLMs
- easy to instruction-tune
- one model can handle many multimodal text-generation tasks

**Weaknesses**
- fluent answers can still be weakly grounded
- projector quality matters a lot
- visual token count can become a major serving bottleneck

**Serving implication**
- highly relevant for latency, TTFT, and KV-cache analysis
- visual tokens directly increase memory pressure and reduce achievable batch size
- this is the most relevant family for serving experiments in this repo

### 4. Query-bridge architectures

**Examples:** BLIP-2, Q-Former-style bridges

Instead of passing all visual features directly into the LLM, these models learn a compact set of query tokens that summarize useful visual information and pass only those to the language model.

**Best for**
- parameter-efficient multimodal adaptation
- lower-cost bridging of strong frozen vision and language components

**Strengths**
- more efficient than end-to-end multimodal retraining
- reduces the amount of visual information pushed into the LLM
- often a strong engineering compromise

**Weaknesses**
- the bottleneck can discard useful visual detail
- less conceptually simple than a direct projector-only design

**Serving implication**
- can reduce token inflation compared with naïvely forwarding many image patches
- a good architecture to discuss when balancing visual fidelity vs serving cost

### 5. Unified encoder-decoder generative models

**Examples:** PaLI, Pix2Struct

These models provide a unified generative interface over visual and textual input. They are often strong when the output is text and the model must parse visual structure or multilingual content.

**Best for**
- image-to-text generation
- multilingual multimodal tasks
- document and screenshot understanding
- structured extraction from visual input

**Strengths**
- one generation interface for many tasks
- strong fit for document-heavy and UI-heavy tasks
- natural output format for extraction and explanation

**Weaknesses**
- more expensive than retrieval-only models
- often not the best choice if retrieval is the main product requirement

**Serving implication**
- sensitive to image resolution and long outputs
- document workloads can create especially high visual-token pressure

### 6. OCR-free document specialists

**Examples:** Donut, Pix2Struct

These models are specialized for documents, forms, screenshots, and UI-like inputs, often avoiding a separate OCR stage.

**Best for**
- invoices and forms
- screenshots and dashboards
- visually grounded extraction
- OCR-free document pipelines

**Strengths**
- simpler end-to-end pipeline than OCR + separate parser
- avoids some OCR error propagation

**Weaknesses**
- may still struggle with very dense layouts or small text
- less explicit intermediate structure than OCR pipelines
- high-resolution inputs remain expensive

**Serving implication**
- document understanding is often more resolution-sensitive than natural-image tasks
- large pages can increase latency sharply

### 7. Interleaved multimodal sequence models

**Examples:** Flamingo, Kosmos-style systems

These are built for richer image-text interleaving rather than a simple "one image, one question" pattern.

**Best for**
- multimodal in-context learning
- assistant-like prompting over multiple images and text turns
- image-text few-shot examples

**Strengths**
- flexible prompting interface
- closer to real assistant behavior

**Weaknesses**
- more complex training and serving behavior
- less straightforward to optimize than simple single-image flows

**Serving implication**
- caching and scheduling become more subtle when image/text turns are interleaved
- realistic for agent-like and assistant-like products

### 8. Encoder-free / pure decoder multimodal models

**Examples:** Fuyu-style architectures

These models avoid a separate vision encoder and feed image patches more directly into a decoder-only transformer.

**Best for**
- research into simpler multimodal stacks
- architectures that want a more unified token-processing story

**Strengths**
- conceptually unified sequence model
- avoids some mismatch between a frozen vision encoder and an LLM

**Weaknesses**
- harder to train well
- less common as the default production recipe

**Serving implication**
- research-interesting, but not the first architecture family to optimize for production

## Quick comparison

| Family | Representative models | Best for | Main weakness | Serving implication |
| :--- | :--- | :--- | :--- | :--- |
| Dual-encoder | CLIP, SigLIP | Retrieval, zero-shot classification | Not naturally generative | Fast embeddings, lower cache pressure |
| Fusion / cross-attention | ALBEF, Flamingo-style | Grounding, VQA, interleaved conditioning | More complex and expensive | Stronger grounding, higher inference cost |
| Vision encoder + projector + LLM | LLaVA-style assistants | Multimodal chat and generation | Can sound fluent without strong grounding | High visual-token and KV-cache pressure |
| Query bridge | BLIP-2 | Parameter-efficient multimodal adaptation | Information bottleneck may lose detail | Lower token inflation than naive projector paths |
| Unified generative encoder-decoder | PaLI, Pix2Struct | Image-to-text, multilingual, document tasks | Heavier than retrieval models | Sensitive to resolution and output length |
| OCR-free document specialist | Donut, Pix2Struct | Documents, screenshots, forms | Dense pages remain expensive | Document inputs can dominate latency |
| Interleaved sequence model | Flamingo, Kosmos-style | Multimodal prompting and context learning | Harder scheduling and serving | More realistic assistant-like serving |
| Encoder-free multimodal decoder | Fuyu-style | Research and architectural simplification | Less established production recipe | Interesting research direction |

## How to choose the right family

Use a **dual encoder** if your main product is retrieval, ranking, or zero-shot classification.

Use a **projector + LLM** or **interleaved assistant model** if your product is a multimodal assistant or image-grounded conversation.

Use a **query-bridge architecture** if you want a better compromise between multimodal capability and serving cost.

Use a **document specialist** or a **unified generative model** if you care about screenshots, forms, PDFs, or document parsing.

## Why VLMs are hard to serve

### Visual token inflation

Images are not processed as single units. They are converted into many visual features or visual tokens. A single image can consume a large chunk of the model context before the model generates any text.

This increases:
- time to first token (TTFT)
- KV-cache memory usage
- latency for long or high-resolution inputs
- risk of smaller effective batch size

### Resolution vs. cost

Higher resolution means better detail and better OCR-like behavior, but it also means more visual tokens, more attention cost, and more memory pressure.

### Document understanding is especially expensive

Tasks such as document OCR, VQA over charts, or screenshot parsing often need fine-grained spatial detail. That frequently makes them more expensive than plain natural-image captioning.

## Which families matter most for this repo

For this project, the most important families are:

1. **Dual encoders** for embedding and retrieval experiments
2. **Vision encoder + projector + LLM** for multimodal serving discussions
3. **Document specialists / Pix2Struct-style models** for document understanding and visual token inflation

Those are the families that connect most directly to:
- batching
- KV-cache behavior
- latency vs throughput
- GPU memory pressure
- serving engines such as vLLM and SGLang

---

## Educational discussion

> **What are the main components of a VLM, and why are they hard to serve?**
>
> A modern generative VLM usually has three parts: a vision encoder, a language model, and an alignment component such as a projector or query bridge. The vision side turns an image into visual tokens, and the language side reasons over those visual tokens together with text tokens.
>
> They are hard to serve because the image already consumes context and memory before text generation even starts. This raises time to first token, increases KV-cache pressure, reduces the usable batch size, and makes high-resolution or document-heavy workloads much more expensive than text-only serving. Optimizing them often requires better batching, cache reuse, and lower-precision execution.
