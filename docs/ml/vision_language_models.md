# Vision-Language Models (VLMs)

A vision-language model jointly processes images and text. The goal is to align visual representations with language
representations so the system can describe, retrieve, classify, ground, or reason over visual content using text.

## Core problem

A VLM must solve two related problems:

1. **Representation alignment**: image and text representations referring to the same concept should be close in a
   shared space
2. **Cross-modal conditioning**: the model should use information from one modality to guide predictions in the other

Typical tasks:

- image-text retrieval
- image captioning
- visual question answering
- grounded conversation over images or videos
- multimodal generation and tool use
- document and screenshot understanding

## Why VLMs are harder than text-only LLMs

A text-only LLM only has to model **one modality**: text.

A VLM must deal with:

- **different input statistics**: images and text have different structures and distributions
- **different encoders**: visual features and language features are not naturally in the same space
- **grounding**: a plausible answer is not enough; the answer must correspond to actual visual evidence
- **serving cost**: images introduce preprocessing, visual encoding, extra tokens, and more memory pressure

This creates two different kinds of difficulty:

### Modeling difficulty

A VLM must learn a bridge between:

- visual features derived from pixels, patches, regions, layout, or objects
- language features derived from tokens, syntax, and semantics

These are not automatically comparable. The model needs either:

- a **shared embedding space**,
- a **fusion mechanism**, or
- a **projection / bridge module**.

### Systems difficulty

Serving a VLM is usually harder than serving a text-only LLM because:

- there is often a **vision encoder** before the LLM decode loop
- there may be many **visual tokens**
- **TTFT** tends to increase
- **KV cache pressure** can increase
- batching and scheduling are more complicated

## Main architecture patterns

## 1) Dual-encoder models

Use one encoder for images and one for text, then align them in a shared embedding space.

A common contrastive objective is

$$
\mathcal{L} = -\sum_i \log \frac{\exp(s(v_i, t_i)/\tau)}{\sum_j \exp(s(v_i, t_j)/\tau)}
$$

with a symmetric text-to-image term as well.

Here:

- $v_i$ is an image embedding
- $t_i$ is a text embedding
- $s(\cdot,\cdot)$ is a similarity score, often cosine similarity
- $\tau$ is a temperature parameter

Representative models:

- CLIP
- SigLIP

```mermaid
flowchart LR
    I[Image] --> IE[Image Encoder]
    T[Text] --> TE[Text Encoder]
    IE --> VI[Image Embedding]
    TE --> VT[Text Embedding]
    VI --> SIM[Similarity in Shared Space]
    VT --> SIM
```

### What this solves

It learns cross-modal retrieval and zero-shot classification by making matched image-text pairs similar and mismatched
pairs dissimilar.

### What CLIP is really doing

CLIP trains:

- an image encoder
- a text encoder

so that **matched image-text pairs** land close together in embedding space, while mismatched pairs are pushed apart.

So yes, conceptually, CLIP learns that:

- an image of a glass
- and a text like "a photo of a drinking glass"

should produce similar vectors.

More precisely:

- it does **not** just align single words to single pixels
- it aligns **whole image semantics** with **whole text semantics**

In training, a batch of image-text pairs is contrasted against itself:

- the correct image-text pairs should score highly
- the wrong pairs should score poorly

### Zero-shot classification

This is one of the most important consequences of CLIP-like training.

#### Intuition

Instead of training a classifier head for each label, you write labels as text prompts, for example:

- "a photo of a cat"
- "a photo of a dog"
- "a photo of a glass"

Then:

1. encode the image
2. encode each text label prompt
3. compare similarities
4. choose the label whose text embedding is closest to the image embedding

So the model performs classification **without task-specific classifier training** for that exact label set.

#### Why it is called zero-shot

Because the model can often classify categories it was **not explicitly trained as a classifier for**.

It has seen large-scale image-text pairs and learned a shared semantic space, so at inference time you can define a new
classification task only through prompts.

#### Example

Suppose the candidate labels are:

- "a photo of a drinking glass"
- "a photo of eyeglasses"
- "a photo of a dog"

Given an image of a drinking glass, the image embedding should be closest to the first text embedding.

That is zero-shot classification in the CLIP sense.

#### Strengths

- efficient retrieval
- clean shared embedding space
- strong zero-shot classification when trained on large image-text corpora
- easy to build retrieval systems and embedding search

#### Weaknesses

- limited fine-grained generative reasoning by itself
- interaction between modalities is weaker than in fully fused architectures
- prompt phrasing can matter a lot
- ambiguity remains hard when the text label is underspecified

## 2) Fusion / cross-attention models

Image features and text features are allowed to interact through cross-attention or multimodal fusion layers.

Representative directions:

- ALBEF-style fusion
- Flamingo-style cross-attention bridges

```mermaid
flowchart LR
    I[Image] --> IE[Vision Encoder]
    T[Text] --> TE[Text Encoder]
    IE --> VF[Visual Features]
    TE --> TF[Text Features]
    VF --> FUSION[Cross-Attention / Fusion Layers]
    TF --> FUSION
    FUSION --> OUT[Task Output]
```

### What this solves

It supports tasks where the text must attend to particular regions or objects, or where fine-grained grounding matters.

### Strengths

- richer multimodal interaction
- better for VQA, grounding, and detailed reasoning
- stronger token-level or region-level interaction than simple shared-space matching

### Weaknesses

- more expensive than simple dual encoders
- usually less retrieval-friendly than a clean shared-space model
- more complex serving path

## 3) Image encoder + projector + LLM

A common modern VLM structure is:

- a **vision encoder** (CNN or ViT) extracts visual tokens or features
- a **projection layer / adapter** maps them into the language model embedding space
- a **decoder-only LLM** consumes both text tokens and visual tokens

Representative models:

- LLaVA
- many multimodal assistant-style LLMs

```mermaid
flowchart LR
    I[Image] --> VE[Vision Encoder]
    VE --> VT[Visual Tokens / Features]
    VT --> P[Projector / Adapter]
    P --> MMT[Projected Visual Tokens]
    T[Prompt Text] --> TOK[Tokenizer]
    TOK --> TXT[Text Tokens]
    MMT --> LLM[Decoder-only LLM]
    TXT --> LLM
    LLM --> OUT[Generated Text]
```

### What this solves

It lets a pretrained language model perform multimodal generation with relatively small vision-specific adaptation.

### Why this is attractive

- reuses the strong reasoning and generation abilities of large language models
- integrates naturally with instruction tuning and conversational interfaces
- supports captioning, VQA, and multimodal dialogue with one core model

### Main tradeoff

The language model may sound coherent even when the visual grounding is weak. Good language fluency is not the same as
faithful visual reasoning.

### Why alignment is different here than in CLIP

In CLIP:

- image embeddings and text embeddings are trained into a **shared comparable space**

In projector + LLM systems:

- the vision features are usually **not directly comparable** to language embeddings at first
- the projector learns how to map the visual representation into the space the LLM can consume

So this is not mainly about nearest-neighbor matching in a shared retrieval space. It is about **conditioning a
generative language model on visual information**.

## 4) Query-bridge architectures

These models compress or query visual information before handing it to the language model.

Representative models:

- BLIP-2
- Q-Former-style bridges

```mermaid
flowchart LR
    I[Image] --> VE[Vision Encoder]
    VE --> VF[Dense Visual Features]
    Q[Learned Query Tokens] --> QB[Query Bridge / Q-Former]
    VF --> QB
    QB --> CQ[Compressed Visual Queries]
    T[Prompt Text] --> TOK[Tokenizer]
    TOK --> TXT[Text Tokens]
    CQ --> LLM[Language Model]
    TXT --> LLM
    LLM --> OUT[Generated Text]
```

### What this solves

It allows strong frozen vision and language components to be connected efficiently, often with fewer trainable
multimodal parameters.

### Strengths

- parameter-efficient adaptation
- lower visual-token pressure than naïvely forwarding many image features
- practical engineering compromise

### Weaknesses

- bottleneck may discard useful detail
- architecture is less direct than a simple projector
- can lose fine-grained information if the bridge is too compressive

## 5) Unified encoder-decoder generative models

These models expose a single text-generation interface over both image and text inputs.

Representative models:

- PaLI
- Pix2Struct

```mermaid
flowchart LR
    I[Image / Document] --> VE[Vision Encoder]
    T[Text Prompt] --> TE[Text / Input Encoder]
    VE --> ENC[Unified Encoder Representation]
    TE --> ENC
    ENC --> DEC[Text Decoder]
    DEC --> OUT[Generated Text / Structured Output]
```

### What this solves

It is useful when the final product is naturally framed as image-conditioned text generation, including multilingual or
document-heavy tasks.

### Strengths

- unified generation interface
- strong fit for multilingual and document tasks
- natural for structured extraction and explanation

### Weaknesses

- more expensive than retrieval-oriented systems
- often not the best choice when retrieval is the core need
- can be heavy for very large-scale interactive serving

## 6) OCR-free document specialists

Representative models:

- Donut
- Pix2Struct

```mermaid
flowchart LR
    D[Document / Screenshot Image] --> VE[Document Vision Encoder]
    VE --> DOC[Layout-aware Visual Representation]
    DOC --> DEC[Text / Structured Decoder]
    DEC --> OUT[Fields / JSON / Natural Language Output]
```

### What this solves

It avoids a separate OCR stage and directly predicts the desired textual or structured output from page images or
screenshots.

### Strengths

- simpler end-to-end document pipeline
- less OCR error propagation
- good fit for invoices, forms, screenshots, and UI parsing

### Weaknesses

- dense pages and tiny text remain hard
- less explicit intermediate structure than OCR + layout systems
- high resolution remains expensive

## 7) Interleaved multimodal sequence models

Representative models:

- Flamingo
- Kosmos-style systems

```mermaid
flowchart LR
    X[Interleaved Image / Text Context] --> MM[Multimodal Sequence Processor]
    MM --> CA[Cross-Attention / Sequence Fusion]
    CA --> DEC[Decoder / Generator]
    DEC --> OUT[Next Tokens / Answer]
```

### What this solves

It supports prompting with richer interleavings of images and text rather than one image followed by one question.

### Strengths

- closer to assistant-like multimodal interaction
- strong few-shot and in-context multimodal prompting story
- flexible prompting format

### Weaknesses

- more complex serving and caching behavior
- harder to optimize operationally
- longer multimodal contexts can be expensive

## 8) Encoder-free / pure decoder multimodal models

Representative direction:

- Fuyu-style systems

```mermaid
flowchart LR
    I[Image Patches / Visual Tokens] --> M[Shared Decoder-only Transformer]
    T[Text Tokens] --> M
    M --> OUT[Generated Text]
```

### What this solves

It pushes toward a more unified token-processing stack by avoiding a separate vision encoder.

### Strengths

- conceptually simple sequence processing story
- avoids some mismatch between an external vision encoder and the LLM
- clean “single decoder” mental model

### Weaknesses

- less established as the default production recipe
- still more of a research and exploration direction for many teams
- training efficiency and quality tradeoffs can be challenging

## Comparison of major architecture families

| Family                          | Core idea                               | Best for                                  | Main weakness                         |
|---------------------------------|-----------------------------------------|-------------------------------------------|---------------------------------------|
| Dual encoder                    | Shared image/text embedding space       | retrieval, zero-shot classification       | weak generative reasoning             |
| Fusion / cross-attention        | richer token-level interaction          | VQA, grounding                            | higher compute                        |
| Projector + LLM                 | map visual tokens into LLM space        | multimodal assistants                     | fluent but weakly grounded answers    |
| Query bridge                    | compress useful visual information      | parameter-efficient multimodal adaptation | information bottleneck                |
| Unified encoder-decoder         | one text-generation interface           | multilingual/document generation          | heavier serving cost                  |
| OCR-free document model         | predict output directly from page image | end-to-end document extraction            | tiny text and dense pages remain hard |
| Interleaved multimodal sequence | images and text mixed in context        | multimodal prompting                      | serving/caching complexity            |
| Encoder-free multimodal decoder | one decoder over all tokens             | unified research direction                | less mature production recipe         |

## Visual tokenization

Images are not fed in as raw pixels to an LLM directly. Common steps are:

1. encode image patches or regions with a vision backbone
2. produce a sequence of visual embeddings
3. project those embeddings into the multimodal token space

This is analogous to text tokenization in spirit: the continuous image is converted into discrete positions or feature
tokens that later modules can process sequentially.

## Training objectives

Different VLMs combine several objectives.

### Contrastive alignment

Match paired image/text items and separate mismatched ones.

Why it is used: excellent for retrieval and shared semantic space learning.

### Image captioning / autoregressive loss

Predict text conditioned on an image:

$$
\log p(y_1,\dots,y_T \mid \text{image}).
$$

Why it is used: teaches fluent image-conditioned generation.

### Matching / binary classification losses

Predict whether an image and a text belong together.

Why it is used: strengthens pairwise compatibility reasoning.

### Instruction tuning

Train on multimodal instruction-response pairs.

Why it is used: turns a base VLM into a useful assistant that follows user prompts over images.

## Typical components

| Component                             | Role                               | Why it is needed                                                                  |
|---------------------------------------|------------------------------------|-----------------------------------------------------------------------------------|
| Vision encoder                        | Extract visual features            | Images have strong spatial structure that is best handled before fusion with text |
| Text encoder or LLM                   | Process language                   | Provides semantic composition and generation                                      |
| Projector / adapter                   | Align feature spaces               | Vision and language embeddings usually have different dimensions and statistics   |
| Cross-attention or multimodal decoder | Fuse modalities                    | Needed when fine-grained conditioning matters                                     |
| Query bridge                          | Compress useful visual information | Helps balance multimodal quality with serving cost                                |

## Main tradeoffs

| Design choice                 | Benefit                                            | Cost                                       |
|-------------------------------|----------------------------------------------------|--------------------------------------------|
| Dual encoder                  | Retrieval-friendly, scalable contrastive training  | Weaker token-level cross-modal interaction |
| Cross-attention fusion        | Better grounding and detailed multimodal reasoning | Higher compute and complexity              |
| LLM-based VLM                 | Strong generation and instruction following        | Risk of fluent but weakly grounded answers |
| Query bridge                  | Lower multimodal adaptation cost                   | Possible information bottleneck            |
| High-resolution visual tokens | Better detail                                      | More memory and slower inference           |
| OCR-free document model       | Simpler end-to-end document stack                  | Dense pages remain expensive               |

## Failure modes

### Hallucination

The model describes objects or relations not actually present.

Why it happens: the language prior can dominate the visual evidence.

### Weak grounding

The answer is plausible but not tied to the correct region or object.

### OCR and fine-detail failure

Small text, charts, tables, or dense scenes can require higher resolution or specialized modules.

### Over-compression of visual context

A compact bridge can reduce serving cost, but if it removes too much visual information, the model may miss details that
matter for grounding or extraction.

## Minimal code sketch

```python
image_tokens = vision_encoder(image)
image_tokens = projector(image_tokens)
text_tokens = tokenizer(prompt)
output = multimodal_llm(text_tokens, image_tokens)
```

## What to remember

- VLMs solve both representation alignment and cross-modal conditioning
- Dual encoders are excellent for retrieval; fused or LLM-based models are better for rich multimodal generation
- CLIP-style zero-shot classification works by comparing an image embedding to candidate label prompts in a shared space
- A modern VLM often combines a vision encoder, an adapter or query bridge, and a language model
- The main quality risk is not fluency but **faithful grounding in the visual input**
- The main systems risk is that more visual detail usually means more memory pressure and slower inference
