# Gradient-Boosted Trees

Gradient-boosted trees build an additive model of many small trees, trained **sequentially** so that each new tree
improves the current ensemble.

This note covers classical GBM, and then explains how important implementations such as **XGBoost**, **LightGBM**, and
**CatBoost** differ in practice.

## The core idea

Instead of fitting one large tree, boosting constructs a predictor of the form

$$
F_M(x) = F_0(x) + \sum_{m=1}^M \eta \, \gamma_m \, h_m(x),
$$

where:

- $F_0(x)$ is an initial prediction
- $h_m(x)$ is a small regression tree at stage $m$
- $\gamma_m$ is the step size or leaf value scale
- $\eta$ is the learning rate

The important point is that trees are added **one after another**, not independently.
Each new tree is trained to improve the loss of the current ensemble.

## What problem boosting is solving

A single shallow tree has high bias.
Boosting reduces that bias by adding many small corrective trees.

Bagging says:

> average many noisy strong learners.

Boosting says:

> build a strong learner by adding weak learners that fix the current errors.

So gradient boosting is best understood as a **stage-wise functional optimization** method.

## General gradient boosting algorithm

Let the model after stage $m-1$ be $F_{m-1}(x)$ and the loss be $\mathcal L(y, F(x))$.

### Step 0: initialize

Choose a constant function

$$
F_0(x) = \arg\min_c \sum_{i=1}^n \mathcal L(y_i, c).
$$

Examples:

- squared error: initialize with the mean target
- logistic loss: initialize with the log-odds of the positive class

### Step 1: compute pseudo-residuals

For each training example, compute the negative gradient of the loss with respect to the current prediction:

$$
r_i^{(m)} = -\left.\frac{\partial \mathcal L(y_i, F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}}.
$$

These are the directions in function space that most reduce the loss.

### Step 2: fit a tree to the pseudo-residuals

Train a small regression tree $h_m(x)$ to predict $r_i^{(m)}$ from $x_i$.

This tree partitions the input space into leaves $R_{jm}$.

### Step 3: compute the best leaf values

For each leaf $R_{jm}$, choose an update value

$$
\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} \mathcal L\bigl(y_i, F_{m-1}(x_i) + \gamma\bigr).
$$

For squared error, this is just the mean residual in the leaf.
For logistic loss, the expression is different but the idea is the same.

### Step 4: update the ensemble

$$
F_m(x) = F_{m-1}(x) + \eta \sum_j \gamma_{jm} \mathbf 1[x \in R_{jm}].
$$

### Step 5: repeat

Repeat the procedure for $m = 1, \dots, M$ until the validation metric stops improving or the budget is exhausted.

## How to interpret the algorithm

A useful picture is:

1. the current model makes predictions
2. we look at where the loss wants those predictions to move
3. we fit a small tree that approximates that desired correction
4. we add only a small step in that direction

That small-step idea is what makes boosting stable enough to work well.

## Squared-error example

For regression with

$$
\mathcal L(y, F(x)) = \frac{1}{2}(y - F(x))^2,
$$

the negative gradient is simply

$$
r_i^{(m)} = y_i - F_{m-1}(x_i),
$$

which is the ordinary residual.

So for squared loss, gradient boosting really does fit trees to residual errors.

## Logistic-loss example

For binary classification, the model usually predicts a score $F(x)$, which maps to a probability through

$$
p(x) = \sigma(F(x)).
$$

The pseudo-residuals become related to the difference between observed labels and current predicted probabilities.
So the next tree is pushed toward the examples that are still badly modeled.

## Why shallow trees are common

Boosting usually uses shallow trees, often depth 3 to 8.
These are weak enough to regularize the stage-wise procedure, but expressive enough to capture interactions.

A depth-1 tree is a stump.
A depth-3 or depth-6 tree can already represent useful feature interactions.

## Main regularization controls

### Learning rate

A smaller learning rate means each tree makes a smaller correction.
This usually requires more trees, but often improves generalization.

### Number of trees

More stages increase capacity.
Too many stages can overfit, especially without early stopping.

### Tree depth / number of leaves

Shallower trees reduce interaction order and help control overfitting.
Deeper trees can model richer interactions, but increase variance.

### Row subsampling

Train each stage on a random subset of examples.
This adds stochasticity and can improve generalization.

### Column subsampling

Use only a subset of features for each tree or split.
This can reduce overfitting and training cost.

### Early stopping

Monitor a validation set and stop when the metric no longer improves.
This is one of the most important practical regularizers.

## Classical GBM vs random forest

- **random forest**: trees are trained independently and averaged; variance reduction is the main story
- **GBM**: trees are trained sequentially to improve the current model; bias reduction plus regularization is the main story

Random forests are usually simpler to tune.
GBMs are often stronger on tabular prediction tasks when tuned carefully.

## How LightGBM works

LightGBM is a highly optimized gradient-boosting implementation designed for large tabular datasets.
The main algorithmic ideas are:

### 1) Histogram-based split finding

Instead of testing every raw threshold, LightGBM buckets continuous features into discrete bins.
Then split search works on bin statistics rather than raw values.

This makes training much faster and more memory-efficient.

### 2) Leaf-wise tree growth

Many classical GBM implementations grow trees level by level.
LightGBM usually grows the **leaf with the largest estimated gain** next.

This often reduces training loss faster because the algorithm focuses capacity where it helps most.
But it can also overfit more easily on small datasets unless constraints such as `num_leaves`, `min_data_in_leaf`, or
`max_depth` are set carefully.

### 3) GOSS (Gradient-based One-Side Sampling)

LightGBM can keep examples with large gradients and subsample more aggressively among examples with small gradients.

Intuition:

- large gradients correspond to examples the model is still getting wrong
- small gradients correspond to examples already fit reasonably well

This keeps informative examples while reducing training cost.

### 4) EFB (Exclusive Feature Bundling)

Sparse features that are rarely active at the same time can be bundled together into a smaller number of composite
features.
This reduces dimensionality and speeds training, especially in sparse settings.

## How XGBoost differs

XGBoost is another major GBM implementation.
Important traits include:

- strong regularization in the tree objective
- efficient approximate split finding
- careful handling of sparsity and missing values
- widespread support and mature tooling

XGBoost often feels like the most general-purpose, highly tunable GBM implementation.

## How CatBoost differs

CatBoost is especially strong when categorical variables matter.
Its main ideas include:

- ordered boosting to reduce target leakage from target statistics
- specialized handling of categorical features
- strong defaults for mixed tabular datasets with many categorical columns

CatBoost is often an excellent choice when feature engineering for categories would otherwise be painful.

## When GBMs are a strong default

Gradient-boosted trees are often the strongest classical default for:

- structured tabular data
- business prediction problems
- ranking tasks
- datasets with nonlinear interactions and heterogeneous feature scales
- medium-to-large datasets where careful validation is possible

## When GBMs are a weaker fit

They are usually less natural for:

- raw text, image, and audio inputs
- extremely small datasets with severe noise and unstable validation
- settings where a single transparent decision path is required
- tasks where online updates or ultra-low-latency inference dominate and ensemble size becomes a problem

## Important hyperparameters

### Shared ideas across implementations

- `learning_rate`
- `n_estimators` / boosting rounds
- tree depth or leaf count
- row subsampling
- column subsampling
- regularization terms
- early stopping patience

### LightGBM-specific knobs

- `num_leaves`
- `min_data_in_leaf`
- `max_depth`
- `feature_fraction`
- `bagging_fraction`
- `lambda_l1`, `lambda_l2`

A common practical rule is that `num_leaves` controls much of the model complexity.

## Small code examples

### Scikit-learn gradient boosting

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    learning_rate=0.05,
    n_estimators=300,
    max_depth=3,
    random_state=0,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

### LightGBM

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
)
```

## What to remember

- Gradient boosting builds an additive model stage by stage.
- Each new tree is trained to reduce the current loss, often via negative gradients.
- Shallow trees, shrinkage, subsampling, and early stopping are the main regularizers.
- LightGBM speeds boosting with histograms and leaf-wise growth; XGBoost and CatBoost provide different optimization
  and categorical-data tradeoffs.
- On tabular prediction tasks, GBMs are often one of the strongest practical defaults.
