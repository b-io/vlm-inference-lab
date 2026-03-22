# Model selection and use cases

This note summarizes **when one model family is better suited than another**, what problem each family is solving, and
the main tradeoffs that drive the choice.

## 1) The selection problem

Choosing a model is usually an optimization over several objectives at once:

- predictive performance
- amount and type of data available
- latency / throughput constraints
- memory footprint
- interpretability
- calibration and uncertainty needs
- robustness to distribution shift
- ease of training and maintenance

There is no universally best model. The practical question is:

> Given the task, the data, and the constraints, which inductive bias is most appropriate?

An **inductive bias** is the set of structural assumptions a model makes about the data. Good model selection is mostly
about matching these assumptions to the problem.

---

## 2) Fast mental model

| Data / task regime                                  | Often strong default                        | Why                                                                                       |
|-----------------------------------------------------|---------------------------------------------|-------------------------------------------------------------------------------------------|
| Small or medium-size **tabular** data               | Gradient-boosted trees                      | Strong performance with little preprocessing; handles nonlinear feature interactions well |
| Tabular data with strict **interpretability** needs | Linear / logistic regression, shallow trees | Easy to inspect coefficients or decision paths                                            |
| High-dimensional sparse text (classical NLP)        | Linear models on bag-of-words / TF-IDF      | Strong baseline, cheap, robust, often hard to beat on small datasets                      |
| Long-range **sequence modeling**                    | Transformers                                | Self-attention models long-range dependencies and parallelizes training                   |
| Short sequences / streaming with tight memory       | RNN / GRU / temporal CNN                    | Lower state size, natural online processing                                               |
| Natural images with translation-local structure     | CNNs                                        | Convolution encodes locality and weight sharing                                           |
| Large-scale vision with enough data / compute       | Vision Transformers                         | Flexible global context modeling, strong scaling behavior                                 |
| Multimodal image+text reasoning                     | VLMs                                        | Joint or aligned vision-language representations                                          |
| Very small datasets                                 | Simpler models, transfer learning           | Lower variance and better sample efficiency                                               |
| High-stakes deployment needing explanation          | Linear models, GAMs, trees                  | Easier to audit and communicate                                                           |

---

## 3) What different model families are trying to exploit

### Linear models

A linear model assumes the prediction is approximately an affine function of the features:

$$
\hat y = w^\top x + b
$$

For classification, logistic regression models class probability as:

$$
P(y=1 \mid x) = \sigma(w^\top x + b)
$$

**What it exploits:** approximate linear separability or additive effects in a chosen feature space.

**When it is better suited:**

- limited data
- many features but weak nonlinear structure
- strong need for interpretability
- sparse high-dimensional features such as TF-IDF

**Tradeoff:** low variance and easy optimization, but limited expressivity unless features are engineered carefully.

---

### Decision trees

A tree partitions the feature space by recursive axis-aligned splits. For classification, a split is chosen to reduce
impurity; for regression, to reduce variance.

**What it exploits:** threshold structure, heterogeneous local rules, and feature interactions.

**When it is better suited:**

- heterogeneous tabular data
- nonlinear effects with simple logical rules
- missing values / mixed feature scales
- need for local explanations

**Tradeoff:** single trees are interpretable but unstable and can overfit.

---

### Random forests

A random forest averages many trees grown on bootstrap samples with feature subsampling.

**What it exploits:** the same tree bias as above, but reduces variance by averaging decorrelated trees.

**When it is better suited than a single tree:**

- tabular problems where variance is a concern
- strong baseline needed with limited tuning
- robustness matters more than interpretability of a single path

**Tradeoff:** better generalization than one tree, but less interpretable and often weaker than gradient boosting on
many structured tabular datasets.

---

### Gradient-boosted trees

Boosting builds trees sequentially to fit residual errors or negative gradients of a loss.

At stage $m$, the new learner approximates the negative gradient:

$$
r_i^{(m)} \approx -\left.\frac{\partial \mathcal L(y_i, F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}}
$$

**What it exploits:** complex nonlinear interactions in tabular data with strong bias-variance control.

**When it is better suited:**

- structured tabular data
- mixed numeric/categorical signals
- moderate dataset sizes
- competition-style predictive accuracy on business data

**Tradeoff:** powerful on tabular tasks, but less natural for raw text, audio, or image inputs.

---

### SVMs

Support Vector Machines maximize margin. In the soft-margin linear case:

$$
\min_{w,b,\xi} \ \frac{1}{2}\lVert w\rVert^2 + C \sum_i \xi_i
$$

subject to:

$$
y_i(w^\top x_i + b) \ge 1 - \xi_i, \qquad \xi_i \ge 0
$$

With kernels, SVMs implicitly operate in a richer feature space.

**When they are better suited:**

- small to medium datasets
- high-dimensional spaces
- carefully engineered feature representations
- need for strong geometric regularization

**Tradeoff:** good theoretical properties, but kernel scaling is difficult on very large datasets.

---

### CNNs

Convolution applies local filters across the input:

$$
y[i,j] = \sum_{u,v,c} K[u,v,c] \, x[i+u, j+v, c]
$$

**What it exploits:** locality, translation structure, and parameter sharing.

**When CNNs are better suited:**

- images where nearby pixels are highly correlated
- limited data relative to model size
- tasks that benefit from spatial hierarchy: edges $\rightarrow$ textures $\rightarrow$ parts $\rightarrow$ objects

**Why better than a generic MLP on images:** an MLP ignores the 2D structure and uses far more parameters; a CNN imposes
the right bias for natural images.

**Tradeoff:** excellent inductive bias for local spatial structure, but less flexible for arbitrary long-range
interactions unless depth/receptive field is increased.

---

### RNN / LSTM / GRU

An RNN maintains a hidden state:

$$
h_t = f(h_{t-1}, x_t)
$$

LSTMs add gates to control memory flow.

**What they exploit:** temporal order and compact recurrent state.

**When they are better suited than Transformers:**

- streaming or online inference
- limited memory budgets
- relatively short sequences
- low-latency stateful processing

**Tradeoff:** sequential computation limits parallelism and long-range dependency handling.

---

### Transformers

Transformers use self-attention:

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**What they exploit:** flexible pairwise interaction between tokens, with short path length between distant positions.

**When they are better suited than RNNs:**

- long-range sequence modeling
- large datasets and pretraining
- tasks needing global context
- training regimes where parallelization matters

**Tradeoff:** quadratic attention cost in sequence length for full attention, higher memory use, and often more
data-hungry.

---

### Vision Transformers

A ViT splits an image into patches, projects patches to tokens, then applies Transformer blocks.

**When ViTs are better suited than CNNs:**

- large datasets or strong pretraining
- tasks benefiting from global context and flexible receptive fields
- unified architectures across modalities

**Tradeoff:** weaker built-in locality bias; typically needs more data or augmentation than CNNs from scratch.

---

### Vision-language models (VLMs)

A VLM couples a vision encoder with a language model or learns aligned image-text embeddings.

**What they exploit:** semantic alignment across modalities.

**When they are better suited:**

- image captioning
- visual question answering
- document understanding
- retrieval across text and images
- agentic systems that need image-grounded reasoning

**Tradeoff:** more complex pipelines, larger compute cost, and multimodal failure modes.

---

## 4) Architecture-oriented comparison

| Model family                 | Best suited for                                          | Why it fits                                            | Main limitations                                             |
|------------------------------|----------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------------|
| Linear / logistic regression | Interpretable baselines, sparse features, small datasets | Low variance, convex optimization, simple coefficients | Misses nonlinear interactions unless features are engineered |
| Decision tree                | Rule-like decision boundaries, local if-then logic       | Naturally models thresholds and interactions           | High variance, unstable                                      |
| Random forest                | Robust tabular baseline                                  | Variance reduction by bagging                          | Larger, less interpretable                                   |
| Gradient-boosted trees       | Structured tabular prediction                            | Very strong nonlinear tabular inductive bias           | Less natural for raw unstructured inputs                     |
| SVM                          | Small/medium high-dimensional datasets                   | Margin maximization, kernels                           | Poor scaling for large datasets                              |
| CNN                          | Vision, local spatial signals                            | Locality + weight sharing                              | Harder to capture global context directly                    |
| RNN/LSTM/GRU                 | Streaming temporal modeling                              | Compact state, natural online updates                  | Sequential bottleneck                                        |
| Transformer                  | Text, long contexts, sequence pretraining                | Global attention, parallel training                    | Attention memory/time cost                                   |
| ViT                          | Large-scale vision                                       | Flexible global interactions                           | Often needs more data/pretraining                            |
| VLM                          | Multimodal reasoning/retrieval                           | Shared or aligned text-image representations           | Complexity and compute                                       |

---

## 5) Use-case patterns

### Tabular business data

Examples: credit risk, churn, fraud features, pricing.

**Usually start with:**

- gradient-boosted trees
- linear/logistic regression as a baseline
- maybe random forest for robustness

**Why not default to a deep NN?**
Tabular data is often low-dimensional, heterogeneous, and sample sizes are moderate. Tree ensembles often dominate
because their inductive bias matches thresholding and feature interactions.

---

### Text classification

Examples: spam detection, topic classification, sentiment.

**Small data / latency-sensitive:**

- TF-IDF + logistic regression or linear SVM

**Large data / transfer learning:**

- pretrained Transformer encoder

**Why the split?**
Sparse linear models are very strong when the task mostly depends on keyword patterns and data is limited. Transformers
become better when context, compositionality, or transfer from large-scale pretraining matters.

---

### Forecasting and temporal signals

Examples: sensor data, logs, demand forecasting.

**Better suited models depend on horizon and regime:**

- simple autoregressive / linear baselines for strong short-term structure
- boosted trees for feature-rich tabularized forecasting
- RNN/TCN for sequential online settings
- Transformers when long context and many interacting signals matter

No single sequential model dominates all time-series problems.

---

### Computer vision

Examples: classification, detection, segmentation.

**Smaller data or strong locality bias needed:**

- CNNs

**Large pretraining or foundation-model setting:**

- ViTs

CNNs are often better when data is limited because locality and weight sharing are useful priors. ViTs are often better
when massive pretraining weakens the need for hand-designed inductive bias.

---

### Multimodal systems

Examples: OCR + reasoning, chart QA, document QA, image-grounded assistants.

**Better suited model:**

- VLM, or a pipeline with a vision encoder plus text model

Why? The task is not only vision or only language; the model must align visual evidence to linguistic reasoning.

---

## 6) How constraints change the answer

### Interpretability

Prefer:

- linear/logistic regression
- generalized additive models
- shallow trees

Avoid using a more complex model unless the performance gain is worth the interpretability loss.

### Latency

Prefer:

- linear models
- shallow trees
- small CNNs
- compact RNNs
- distilled Transformers

Full attention Transformers and large VLMs can be too expensive for edge or real-time systems.

### Data scarcity

Prefer:

- simpler models
- transfer learning
- hand-crafted features where appropriate

High-capacity models without enough data tend to overfit.

### Distribution shift

Models with strong pretraining or simpler decision boundaries may generalize better, but domain shift must be tested
empirically. Ensembles and calibration often matter as much as architecture choice.

---

## 7) Practical selection workflow

1. Define the task and metric clearly.
2. Start with a **simple baseline**.
3. Match model bias to the data modality.
4. Prefer the simplest model that meets the metric and deployment constraints.
5. Use ablations:
    - better architecture?
    - better features?
    - more data?
    - transfer learning?
6. Evaluate calibration, latency, interpretability, and robustness, not only accuracy.

---

## 8) Minimal code sketch

```python
# A minimal model-selection loop in scikit-learn style
candidates = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=300, random_state=0),
    "xgb_like": HistGradientBoostingClassifier(random_state=0),
}

for name, model in candidates.items():
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_valid)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_valid)
    # compute the metric relevant to the problem
```

The main point is not the code. It is that model selection is always task- and constraint-dependent.

---

## 9) What to remember

- Prefer a model whose **inductive bias** matches the data.
- On tabular data, tree ensembles are often hard to beat.
- On text and long sequences, Transformers dominate when enough data and compute are available.
- On images, CNNs encode locality well; ViTs become very strong with scale and pretraining.
- Simpler models are often better under limited data, strict latency, or strong interpretability requirements.

## 10) Related notes

For a more architecture-first summary across MLPs, CNNs, RNNs, Transformers, GNNs, diffusion backbones, and MoE,
see [Neural architecture tradeoffs](../architectures/neural_architecture_tradeoffs.md).
