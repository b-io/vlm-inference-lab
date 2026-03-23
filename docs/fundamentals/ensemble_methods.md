# Ensemble Methods

Ensemble methods are the **umbrella theory** for combining multiple models so that the final predictor is usually
**more accurate, more stable, or more robust** than any single member.

A useful hierarchy is:

- **decision tree**: a single base learner
- **ensemble method**: a strategy for combining multiple learners
- **random forest / gradient-boosted trees / stacking**: specific ensemble algorithms

So decision trees are often the **building block**, while ensemble methods are the **higher-level idea**.

This note connects to:

- [Decision trees](decision_trees.md)
- [Random forests](random_forests.md)
- [Gradient-boosted trees](gradient_boosted_trees.md)
- [Model evaluation and generalization](model_evaluation_and_generalization.md)
- [Model selection and use cases](model_selection_and_use_cases.md)

## Why ensembles work

Suppose we average $B$ predictors with the same variance $\sigma^2$ and pairwise correlation $\rho$.
Then the variance of the average is approximately

$$
\mathrm{Var}\left(\frac{1}{B}\sum_{b=1}^B f_b(x)\right)
= \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2.
$$

This equation explains most of the story:

- if the base learners are **high variance**, averaging helps
- if their errors are **less correlated**, averaging helps much more
- as $B$ grows, the independent part of the variance shrinks

So ensemble design is mostly about two questions:

1. how do we create strong base learners?
2. how do we make their errors different enough that combining them helps?

## Top-level taxonomy

The main ensemble families are:

1. **averaging / voting**: train models independently and combine their predictions
2. **bagging**: resample the data, train many learners independently, then average or vote
3. **boosting**: train learners sequentially so each new learner corrects the current ensemble
4. **stacking**: train a meta-model on the predictions of several base models

In tabular machine learning, the most important tree-based ensemble methods are usually:

- [random forests](random_forests.md) for bagging-based variance reduction
- [gradient-boosted trees](gradient_boosted_trees.md) for stage-wise boosting
- implementation families such as **XGBoost**, **LightGBM**, and **CatBoost** within the gradient-boosting family

## Main ensemble families

### 1) Averaging / voting

Train several models independently, then combine predictions.

- **regression**: average predictions
- **classification**: majority vote or average class probabilities

This is the simplest ensemble. It works when different models make partially different mistakes.

Typical examples:

- average several neural-network checkpoints
- vote across different classifiers
- average multiple folds or seeds at inference time

### 2) Bagging

**Bagging** means **bootstrap aggregating**:

1. sample a bootstrap dataset by drawing $n$ training examples with replacement
2. train a base learner on that sample
3. repeat many times
4. average or vote across the trained learners

Bagging mostly helps by **reducing variance**.
It is most useful for unstable learners, especially deep decision trees.

Common members of the bagging family:

- **bagged trees**: many deep trees trained on bootstrap samples
- [**random forests**](random_forests.md): bagged trees plus random feature subsampling
- **extra trees / extremely randomized trees**: push tree randomization further by using more random split selection

See also: [Random forests](random_forests.md).

### 3) Boosting

Boosting builds learners **sequentially**, not independently.
Each new learner focuses on what the current ensemble still gets wrong.

Broadly, the algorithmic pattern is:

1. start with a simple initial predictor
2. measure current errors under a chosen loss
3. fit a new weak learner to correct those errors
4. add the learner to the ensemble with a step size or weight
5. repeat until validation performance stops improving

Common members of the boosting family:

- **AdaBoost**: upweights previously misclassified examples
- **Gradient Boosting Machine (GBM)**: fits each learner to the negative gradient or residual of the loss
- **XGBoost**: highly optimized regularized gradient boosting
- **LightGBM**: histogram-based gradient boosting, often fast and memory-efficient on large tabular data
- **CatBoost**: gradient boosting designed to work especially well with categorical features

Boosting often reduces **bias** first, while regularization is used to control variance.
It is usually not described as a pure variance-reduction method the way bagging is.

See: [Gradient-boosted trees](gradient_boosted_trees.md).

### 4) Stacking

Stacking trains several base models, then trains a **meta-model** on their predictions.

Typical workflow:

1. train multiple base learners
2. generate out-of-fold predictions from each one
3. use those predictions as features for a second-level model
4. predict with the meta-model

Stacking can work very well, but it is easier to get wrong because leakage is a real risk.

### 5) Blending

Blending is a simpler relative of stacking:

1. train multiple base models
2. hold out a validation set
3. train a small combiner on the validation predictions

It is operationally simpler than full stacking, but it uses data less efficiently because the held-out set is not used for
base-model fitting.

## Tree ensembles as a special case

Decision trees are not themselves ensemble methods, but they are among the most important **base learners** for
ensembles.

That gives a useful mental picture:

- **decision tree**: one tree
- **random forest**: many trees trained independently with bagging and feature subsampling
- **gradient-boosted trees**: many trees added sequentially to reduce current error

That is why tree ensembles dominate so much classical tabular ML.

## Bagging vs boosting

| Method   | How models are built | Main effect | Typical base learner | Main risk |
|----------|----------------------|-------------|----------------------|-----------|
| Bagging  | Independently        | Variance reduction | Deep trees | Limited gain if models remain highly correlated |
| Boosting | Sequentially         | Bias reduction + controlled variance | Shallow trees | Overfitting if the ensemble becomes too complex |

A simple summary:

- **bagging** says: “train many noisy strong learners and average them”
- **boosting** says: “train learners in sequence so each one corrects earlier mistakes”

## Why trees are such common base learners

Decision trees are a natural fit for ensembles because they are:

- flexible and nonlinear
- able to capture interactions automatically
- cheap to evaluate
- unstable enough that bagging helps a lot
- weak enough in shallow form that boosting can improve them stage by stage

## When each ensemble style is useful

### Averaging / voting

Use when you want:

- a simple accuracy boost from multiple reasonably different models
- a low-risk way to combine complementary predictors
- an easy ensemble that does not change the base training pipelines very much

### Bagging / random forest

Use when you want:

- a strong tabular baseline
- low tuning burden
- robustness
- decent out-of-the-box performance
- a safer choice than one deep tree

### Gradient boosting

Use when you want:

- strong predictive performance on tabular data
- flexible nonlinear interactions
- careful bias-variance control via depth, learning rate, regularization, and early stopping
- competition-grade accuracy on structured business data

### Stacking / blending

Use when you have:

- multiple complementary model families
- enough data to generate reliable out-of-fold meta-features
- the operational maturity to manage a more complex training/inference pipeline

## Common pitfalls

- Averaging highly correlated models gives less gain than expected.
- Bagging does not magically fix strong bias.
- Boosting can overfit noisy labels or leakage if regularization is weak.
- Stacking can leak target information if out-of-fold predictions are not generated correctly.
- Bigger ensembles are not always better if latency or interpretability matters.

## What to remember

- Ensemble methods are the umbrella theory; decision trees are often one of the base learners used inside them.
- Bagging mostly reduces variance; boosting mainly improves fit stage by stage.
- Random forests are bagged, decorrelated decision trees.
- Gradient-boosted trees are sequential additive models optimized against a loss.
- On tabular data, tree ensembles are often the strongest practical default.
