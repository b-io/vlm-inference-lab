# Feature engineering and data representations

This note explains how raw inputs are transformed into features or tensors that a model can use, what strategies are common, and when manual feature engineering is still important.

## 1) The core problem

A model usually consumes vectors, tensors, token IDs, or graph structures, not raw business records or free-form media.

Feature engineering answers:

> How should raw data be represented so that the model can learn the relevant structure?

A representation should ideally be:
- informative
- stable
- numerically well-behaved
- compatible with the chosen model
- cheap enough to compute in training and serving

---

## 2) Two broad paradigms

### Classical feature engineering
Humans design transformations such as:
- scaling
- one-hot encoding
- buckets / splines
- lag features
- polynomial interactions
- TF-IDF
- image descriptors

This is still strong for:
- tabular data
- small datasets
- interpretable systems
- high-signal domain heuristics

### Learned representations
A neural network learns features automatically from raw or lightly processed inputs:
- token embeddings
- convolutional features
- latent vectors
- multimodal embeddings

This is strong when:
- data is large
- raw structure matters
- manual features are too brittle or incomplete

In practice, real systems often combine both.

---

## 3) Mathematical view

Raw input \(x\) is often transformed by a feature map \(\phi\):

$$
x \mapsto \phi(x) \in \mathbb R^d
$$

Then the model operates on \(\phi(x)\) instead of the original input.

Examples:
- polynomial regression: \(\phi(x) = [1, x, x^2, x^3]\)
- one-hot category encoding
- TF-IDF vector for a document
- learned embedding lookup for a token ID

The feature map can be:
- hand-designed
- data-driven
- learned jointly with the model

---

## 4) Common strategies by data type

## Numeric features

### Standardization
For many models, especially gradient-based ones, it helps to center and scale features:

$$
x' = \frac{x - \mu}{\sigma}
$$

Why:
- more balanced optimization
- prevents one feature from dominating due to units alone
- often important for linear models, SVMs, neural nets

### Min-max scaling
Maps a feature to a bounded interval:

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

Useful when a bounded range is meaningful, but sensitive to outliers.

### Log transforms
For skewed positive variables:

$$
x' = \log(1 + x)
$$

Useful for counts, money, traffic, or any heavy-tailed positive feature.

### Bucketing / binning
Convert a continuous feature into intervals.

Useful when:
- threshold effects are important
- monotonic trends are weak but local regimes matter
- the downstream model benefits from discretized patterns

---

## Categorical features

### One-hot encoding
For a categorical variable with \(K\) possible values, map it to a vector in \(\{0,1\}^K\) with one active coordinate.

Good for:
- low to moderate cardinality
- linear models
- tree models can sometimes use category-aware handling directly

Tradeoff:
- high cardinality leads to wide sparse vectors

### Ordinal encoding
Map categories to integers.

Use only when there is a real order, such as:
- small < medium < large

Otherwise it injects a fake geometry.

### Target / mean encoding
Replace a category by a target statistic, such as the mean target value for that category.

Useful for high-cardinality categories, but can leak label information if not done carefully. Must be fit with out-of-fold logic.

### Learned embeddings
Map each category ID \(i\) to a learned dense vector \(e_i \in \mathbb R^d\).

Good for:
- high-cardinality features
- neural networks
- cases where categories have latent similarity structure

---

## Text

### Bag-of-words / n-grams
Represent a document by token counts or n-gram counts.

### TF-IDF
Term frequency times inverse document frequency downweights globally common words:

$$
\mathrm{tfidf}(t,d) = \mathrm{tf}(t,d)\cdot \log\frac{N}{\mathrm{df}(t)}
$$

Why useful:
- strong sparse baseline
- cheap and effective on classification / retrieval
- interpretable token weights

### Tokenization + embeddings
A tokenizer maps text to token IDs, then an embedding table maps IDs to vectors:

$$
e_t = E[t]
$$

where \(E \in \mathbb R^{V \times d}\).

This is the standard entry point for neural language models.

### Sentence / document embeddings
A model maps a full sequence to a dense vector useful for retrieval, clustering, or downstream models.

---

## Time series

### Lags
Create past-value features such as:

$$
x_{t-1}, x_{t-2}, \dots, x_{t-k}
$$

### Rolling statistics
Examples:
- rolling mean
- rolling std
- rolling min / max
- exponentially weighted averages

### Calendar features
Examples:
- day of week
- month
- holiday indicator
- hour of day

### Frequency-domain features
Examples:
- Fourier terms
- spectral power
- seasonal decomposition summaries

These are useful when periodicity matters.

---

## Images

### Classical preprocessing
- resize
- center crop
- normalization by channel mean/std
- augmentation: flips, crops, color jitter

### Hand-crafted features (older pipeline)
- SIFT, HOG, edge descriptors

These were historically important, but deep CNN features have largely replaced them in mainstream vision tasks.

### Learned vision representations
Images are converted to tensors of shape \(C \times H \times W\), then a CNN or ViT learns features automatically.

For a CNN, the feature extractor is learned through convolutional layers.
For a ViT, the image is split into patches and patch embeddings are fed to a Transformer.

---

## Audio

Common transformations:
- waveform normalization
- spectrograms
- mel spectrograms
- MFCCs

These convert raw time-domain audio into a representation where local frequency structure is easier to model.

---

## Graphs / structured relational data

Possible features:
- node degree
- centrality
- edge attributes
- learned node embeddings
- message-passing features in GNNs

The key point is that representation must preserve the relational structure, not flatten it blindly.

---

## 5) Why feature engineering still matters with neural networks

Even when deep models learn features, the input pipeline still matters:

- normalization affects optimization
- tokenization determines the discrete vocabulary seen by the model
- augmentation determines invariances the model learns
- missing-value handling affects stability
- positional or temporal features may need to be injected explicitly
- structured business signals often benefit from engineered features even in hybrid deep models

Neural networks reduce manual feature design, but they do not remove the need for good representation design.

---

## 6) Important design axes

| Axis | Question | Examples |
|---|---|---|
| Scale | Are numerical ranges comparable? | standardization, normalization |
| Sparsity | Should the representation be sparse or dense? | one-hot vs embeddings |
| Invariance | Which changes should not matter? | image translation, case-folding in text |
| Locality | Is nearby structure important? | convolutions, n-grams, lag windows |
| Order | Does sequence position matter? | positional encoding, time index |
| Cardinality | How many categories or tokens exist? | one-hot, hashing, embeddings |
| Missingness | Is absence informative? | missing indicators, imputation |
| Leakage risk | Does the transform use target info incorrectly? | target encoding without out-of-fold fitting |
| Interpretability | Must humans inspect the features? | buckets, monotonic transforms, explicit ratios |

---

## 7) Common transformations

### Interaction features
Create products or cross terms:

$$
\phi(x_1, x_2) = [x_1, x_2, x_1x_2]
$$

Useful when the model is linear but the effect is not additive.

### Polynomial features
Expand numeric variables to higher-order terms. Useful for smooth nonlinearities in simple models, but can explode dimensionality.

### Hashing trick
Map tokens/categories to a fixed-dimensional space via hashing. Useful for very large vocabularies or streaming systems.

### Missing-value indicators
Create a binary feature showing whether a value was missing. Often useful because missingness itself can carry signal.

### Domain ratios and aggregations
Examples:
- debt / income
- clicks / impressions
- average spend over last 30 days

These encode prior domain knowledge explicitly.

---

## 8) Classical vs learned representations

| Approach | Strength | Weakness | Good use cases |
|---|---|---|---|
| Hand-crafted features | Data-efficient, interpretable, fast | Can miss complex structure | tabular, low-data, regulated settings |
| Learned embeddings | Captures latent similarity, compact | Needs data and training | NLP, recommender systems, high-cardinality categories |
| End-to-end learned raw-input features | Powerful and flexible | Compute-heavy, data-hungry | vision, speech, large language models |
| Hybrid | Best of both in many real systems | More pipeline complexity | production systems with mixed modalities |

---

## 9) Minimal examples

### Numeric + categorical preprocessing in scikit-learn

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

num_cols = ["age", "income"]
cat_cols = ["segment", "country"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ]
)

model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000)),
])
```

### Token IDs to embeddings in PyTorch

```python
import torch
import torch.nn as nn

vocab_size = 50000
d_model = 768

embedding = nn.Embedding(vocab_size, d_model)
token_ids = torch.tensor([[12, 98, 415]])
x = embedding(token_ids)   # shape: [batch, seq_len, d_model]
```

---

## 10) Failure modes

Common mistakes:
- scaling train and test with different statistics
- target leakage in encoders or aggregations
- one-hot exploding dimension for extreme cardinality
- ordinal-encoding unordered categories
- forgetting missing-value handling
- assuming a neural net will fix a bad representation automatically
- mismatching representation to model: e.g. raw IDs into a linear model with no embedding or one-hot step

---

## 11) Practical workflow

1. Understand the modality and task.
2. Define the representation compatible with the model family.
3. Apply basic hygiene:
   - missing values
   - scaling where needed
   - leakage-safe fitting
4. Add domain-informed features if data is limited.
5. Compare:
   - simple engineered features
   - learned features
   - hybrid approaches
6. Keep the transform reproducible in serving.

---

## 12) What to remember

- Feature engineering is the design of the map from raw data to model-usable representation.
- Good features improve both optimization and generalization.
- Classical ML relies heavily on explicit feature maps.
- Deep learning learns many internal features, but input representation design still matters.
- The correct representation depends on modality, data size, model choice, and deployment constraints.
