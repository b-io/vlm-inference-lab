# Vision Models

Vision models learn representations from images or videos. The core structural prior is that nearby pixels are more related than distant pixels, and that patterns such as edges, corners, and textures recur across locations.

## Problem formulation

Given an image $x \in \mathbb{R}^{H \times W \times C}$, common tasks are:
- **classification**: predict a class label
- **detection**: localize and classify objects
- **segmentation**: predict labels for pixels or regions
- **representation learning**: produce embeddings for retrieval or downstream tasks

The central question is how to exploit spatial structure efficiently.

## Convolutional Neural Networks (CNNs)

A convolution applies a small kernel across spatial positions with shared weights.

For a 2D input, one output channel is:
$$
y_{i,j} = \sum_{u,v,c} K_{u,v,c} \, x_{i+u, j+v, c}.
$$

### What convolution tries to solve

A dense layer over all pixels would ignore image locality and require too many parameters. Convolution imposes two useful assumptions:
- **local connectivity**: local neighborhoods matter most for local patterns
- **weight sharing**: the same detector should work at many image positions

### Why CNNs work well for vision

- far fewer parameters than fully connected image models
- translation-sensitive local feature extraction
- hierarchical composition: edges $\to$ textures $\to$ parts $\to$ objects
- strong inductive bias for natural images

### Typical ingredients

- **convolution**: local feature extraction
- **nonlinearity**: usually ReLU-like activations
- **pooling or striding**: reduce resolution and enlarge receptive field
- **normalization**: stabilize optimization

## Receptive field

The receptive field of a unit is the region of the input that can affect it.

Why it matters:
- shallow small-kernel models capture only local detail
- deeper models or downsampling increase access to larger structures
- many vision tasks require both local detail and global context

## Pooling and striding

Pooling or strided convolutions downsample feature maps.

### What they try to solve

- reduce computation and memory
- increase receptive field
- provide a degree of local invariance

### Tradeoff

Downsampling improves efficiency and abstraction, but loses precise spatial detail. This matters for dense prediction tasks like segmentation.

## Residual networks

Residual blocks learn perturbations of the identity:
$$
y = x + F(x).
$$

### What they try to solve

Deep CNNs become hard to optimize. Residual connections provide shorter gradient paths and make very deep models train reliably.

### Why they matter

They allow deeper architectures without severe degradation in optimization quality.

## Vision Transformers (ViTs)

A Vision Transformer splits an image into patches, embeds each patch as a token, adds positional information, and applies Transformer layers.

If the patch size is $P \times P$, the number of patches is approximately
$$
N = \frac{H}{P}\frac{W}{P}.
$$

### What ViTs try to solve

CNNs bake in locality strongly. ViTs relax that bias and use attention to model longer-range interactions more directly.

### Why they can work well

- flexible global context modeling
- strong scaling behavior with data and model size
- natural compatibility with multimodal token-based architectures

### Main tradeoff

ViTs usually need more data or stronger pretraining to match CNN sample efficiency on moderate datasets, because they impose a weaker spatial inductive bias.

## CNN vs ViT

| Model | Strong prior | Main advantage | Main limitation |
|---|---|---|---|
| CNN | Locality + translation-related weight sharing | Sample-efficient and efficient on images | Global interactions are indirect unless architecture is enlarged |
| ViT | Much weaker image-specific prior | Direct global interactions, strong scaling with pretraining | More data hungry, attention cost can be high |

## Detection and segmentation

### Detection

The task is to predict bounding boxes and classes.

Typical architectural idea:
- backbone extracts features
- neck/fusion stage aggregates multi-scale features
- detection head predicts boxes and classes

Why multi-scale features matter: objects vary strongly in size.

### Segmentation

The task is to label pixels or regions.

Architectures often use encoder-decoder structures with skip connections so the model combines coarse semantic information with fine spatial detail.

## Video models

Videos add a temporal axis. Common options:
- 3D CNNs over space and time
- CNN backbone + temporal model
- Transformer-style spatiotemporal attention

The main challenge is balancing temporal context against cost.

## Why ReLU-like activations are common in CNNs

CNNs apply the same feature detector at many spatial locations. ReLU is attractive because it is cheap, preserves gradient flow on the positive side, and promotes sparse feature responses. This often matches the intuition that a feature detector should activate strongly only where a pattern is present.

## Minimal code sketches

### CNN classifier

```python
x = conv(x)
x = relu(x)
x = pool(x)
x = conv(x)
x = relu(x)
logits = linear(flatten(x))
```

### ViT-style model

```python
patches = patch_embed(image)
x = patches + pos_embed
for block in transformer_blocks:
    x = block(x)
logits = classifier(x[:, 0])
```

## What to remember

- CNNs solve vision by exploiting locality and weight sharing
- Deeper CNNs increase receptive field and abstraction
- Residual connections make very deep vision models train well
- ViTs treat image patches as tokens and trade stronger priors for more flexible global modeling
- Vision architecture choice is largely a question of **inductive bias, scale, and compute budget**
