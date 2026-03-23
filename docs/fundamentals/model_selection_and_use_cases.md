# Model selection and use cases

This note summarizes **when one model family is better suited than another**, what problem each family is solving, and
the main tradeoffs that drive the choice.

Related notes:

- [Linear models and losses](linear_models_and_losses.md)
- [Decision trees](decision_trees.md)
- [Ensemble methods](ensemble_methods.md)
- [Random forests](random_forests.md)
- [Gradient-boosted trees](gradient_boosted_trees.md)

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

| Data / task regime                                  | Often strong default                                                                            | Why                                                                                       | Documentation                                                                                                                                              |
|-----------------------------------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Small or medium-size **tabular** data               | [Gradient-boosted trees](gradient_boosted_trees.md)                                             | Strong performance with little preprocessing; handles nonlinear feature interactions well | [GBM / LightGBM / XGBoost / CatBoost](gradient_boosted_trees.md)                                                                                           |
| Tabular data with strict **interpretability** needs | [Linear / logistic regression](linear_models_and_losses.md), [shallow trees](decision_trees.md) | Easy to inspect coefficients or decision paths                                            | [Linear models](linear_models_and_losses.md), [Decision trees](decision_trees.md)                                                                          |
| High-dimensional sparse text (classical NLP)        | [Linear models](linear_models_and_losses.md) on bag-of-words / TF-IDF                           | Strong baseline, cheap, robust, often hard to beat on small datasets                      | [Linear models and losses](linear_models_and_losses.md)                                                                                                    |
| Long-range **sequence modeling**                    | [Transformers](../transformers/transformers_tokenization_and_decoding.md)                       | Self-attention models long-range dependencies and parallelizes training                   | [Transformers, tokenization, and decoding](../transformers/transformers_tokenization_and_decoding.md)                                                      |
| Short sequences / streaming with tight memory       | [RNN / GRU / temporal CNN](../architectures/sequential_models.md)                               | Lower state size, natural online processing                                               | [Sequential models](../architectures/sequential_models.md), [RNN/LSTM/GRU and gradient stability](../architectures/rnn_lstm_gru_and_gradient_stability.md) |
| Natural images with translation-local structure     | [CNNs](../architectures/vision_models.md)                                                       | Convolution encodes locality and weight sharing                                           | [Vision models](../architectures/vision_models.md)                                                                                                         |
| Large-scale vision with enough data / compute       | [Vision Transformers](../architectures/vision_models.md)                                        | Flexible global context modeling, strong scaling behavior                                 | [Vision models](../architectures/vision_models.md)                                                                                                         |
| Multimodal image+text reasoning                     | [VLMs](../multimodal/vlm_architectures_and_basics.md)                                           | Joint or aligned vision-language representations                                          | [VLM architectures and basics](../multimodal/vlm_architectures_and_basics.md)                                                                              |
| Very small datasets                                 | [Simpler models](linear_models_and_losses.md), transfer learning                                | Lower variance and better sample efficiency                                               | [Linear models](linear_models_and_losses.md), [Decision trees](decision_trees.md)                                                                          |
| High-stakes deployment needing explanation          | [Linear models](linear_models_and_losses.md), GAMs, [trees](decision_trees.md)                  | Easier to audit and communicate                                                           | [Linear models](linear_models_and_losses.md), [Decision trees](decision_trees.md)                                                                          |

A tabular-data shortcut that often works well in practice is:

1. start with a simple baseline such as linear/logistic regression
2. try a [random forest](random_forests.md) when you want a robust low-tuning ensemble
3. try [gradient-boosted trees](gradient_boosted_trees.md) when predictive accuracy is the main priority

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

See also: [Linear models and losses](linear_models_and_losses.md).

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

See also: [Decision trees](decision_trees.md).

---

### Ensemble methods

An ensemble combines several models so the final predictor is better than a single model.
The umbrella theory is [ensemble methods](ensemble_methods.md), and the most important ensemble families for tabular ML
are:

- **averaging / voting**: combine independently trained predictors
- **bagging**: train models independently on bootstrap samples, then average
- **random forests**: bagged decision trees with feature subsampling
- **boosting**: train models sequentially so each one corrects current errors
- **stacking**: train a meta-model on the predictions of several base learners

**Why ensembles help:** if the component models are reasonably accurate and not perfectly correlated, aggregation can
reduce variance or improve the bias-variance tradeoff.

See also: [Ensemble methods](ensemble_methods.md).

---

### Random forests

A random forest averages many trees grown on bootstrap samples with feature subsampling.

**What it exploits:** the same tree bias as a single decision tree, but with much better generalization through bagging
and decorrelation.

**How the algorithm works:**

1. draw a bootstrap sample for each tree
2. grow a tree on that sample
3. at every split, only consider a random subset of candidate features
4. repeat for many trees
5. aggregate predictions by averaging or voting

**Why bagging helps:** averaging many unstable trees reduces variance; feature subsampling makes the trees less
correlated, which makes the averaging more effective.

**When it is better suited than a single tree:**

- tabular problems where variance is a concern
- strong baseline needed with limited tuning
- robustness matters more than interpretability of a single path
- you want a safe default before spending time on heavier tuning

**Tradeoff:** better generalization than one tree, but less interpretable and often weaker than gradient boosting on
many structured tabular datasets.

See also: [Random forests](random_forests.md).

---

### Gradient-boosted trees

Boosting builds trees sequentially to fit residual errors or negative gradients of a loss.

At stage $m$, the new learner approximates the negative gradient:

$$
r_i^{(m)} \approx -\left.\frac{\partial \mathcal L(y_i, F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}}
$$

**What it exploits:** complex nonlinear interactions in tabular data with strong bias-variance control.

**How the algorithm works:**

1. initialize with a constant prediction
2. compute pseudo-residuals or negative gradients under the current model
3. fit a small regression tree to those residual targets
4. add the new tree to the ensemble with a small learning rate
5. repeat until the validation metric stops improving

**Where LightGBM fits:** LightGBM is an optimized GBM implementation that uses histogram-based split finding and usually
leaf-wise tree growth, which can make training faster and more memory-efficient on large tabular datasets.

**When it is better suited:**

- structured tabular data
- mixed numeric/categorical signals
- moderate-to-large dataset sizes
- competition-style predictive accuracy on business data
- settings where [LightGBM](gradient_boosted_trees.md), XGBoost, or CatBoost are strong practical choices

**Tradeoff:** powerful on tabular tasks, but less natural for raw text, audio, or image inputs.

See also: [Gradient-boosted trees](gradient_boosted_trees.md).

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

See also: [Vision models](../architectures/vision_models.md).

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

See also: [Sequential models](../architectures/sequential_models.md) and
[RNN, LSTM, GRU, and gradient stability](../architectures/rnn_lstm_gru_and_gradient_stability.md).

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

See also: [Transformers, tokenization, and decoding](../transformers/transformers_tokenization_and_decoding.md).

---

### Vision Transformers

A ViT splits an image into patches, projects patches to tokens, then applies Transformer blocks.

**When ViTs are better suited than CNNs:**

- large datasets or strong pretraining
- tasks benefiting from global context and flexible receptive fields
- unified architectures across modalities

**Tradeoff:** weaker built-in locality bias; typically needs more data or augmentation than CNNs from scratch.

See also: [Vision models](../architectures/vision_models.md).

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

See also: [VLM architectures and basics](../multimodal/vlm_architectures_and_basics.md).

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

- [gradient-boosted trees](gradient_boosted_trees.md)
- [linear/logistic regression](linear_models_and_losses.md) as a baseline
- maybe a [random forest](random_forests.md) for robustness

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
- RNNs, temporal CNNs, or Transformers when sequence structure itself matters strongly

**Why no single winner?**
Temporal tasks differ in whether the key structure lies in lag features, latent state dynamics, long-range dependency,
seasonality, or exogenous signals.

---

### Image recognition

Examples: classification, detection, segmentation.

**Usually start with:**

- CNNs if data is moderate and strong locality bias helps
- ViTs if pretraining and scale are available

**Reason:** the spatial structure of images is the dominant inductive-bias driver.

---

### Multimodal reasoning

Examples: image captioning, VQA, document QA.

**Usually start with:**

- a pretrained VLM or an image encoder + language model stack

**Why?**
The core problem is alignment between vision and language representations, not just classification within one modality.

---

## 6) Selection by constraints, not just task

Sometimes the model family is chosen less by the raw task and more by operational constraints.

### If interpretability dominates

Prefer:

- linear models
- GAMs
- shallow trees

### If latency dominates

Prefer:

- small linear models
- small tree ensembles
- small CNNs / distilled Transformers depending on modality

### If data is scarce

Prefer:

- simpler models with strong priors
- pretrained / transfer learning approaches
- regularized classical models before very large networks from scratch

### If compute is abundant and large pretraining exists

Prefer:

- Transformer-family models for language and multimodal tasks
- ViTs for vision

---

## 7) A practical workflow for selection

A good workflow is often:

1. start with the simplest credible baseline
2. match architecture to modality and structure
3. evaluate under the real metric
4. compare error types, not just aggregate score
5. choose the simplest model that meets the operational goal

Examples:

- for tabular data: logistic regression $\rightarrow$ random forest $\rightarrow$ gradient boosting
- for text: bag-of-words logistic regression $\rightarrow$ pretrained Transformer
- for vision: pretrained CNN baseline $\rightarrow$ ViT if scale justifies it

---

## 8) Small code sketch

```python
# Not a universal recipe; just a workflow sketch.

candidates = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=300),
    "gbt": HistGradientBoostingClassifier(),
}

for name, model in candidates.items():
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
    print(name, score)
```

The main point is not the code. It is that model selection is always task- and constraint-dependent.

---

## What to remember

- Model selection is about matching inductive bias to data structure and operational constraints.
- Simpler models win surprisingly often when data is limited or interpretability matters.
- On tabular data, tree ensembles are often hard to beat.
- Random forests are the main bagging-style default; gradient boosting and LightGBM-style methods are often the main
  accuracy-first defaults.
- Transformers, CNNs, ViTs, and VLMs win when the modality and scale match their architectural assumptions.
