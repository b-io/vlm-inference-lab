# Model Evaluation, Bias-Variance, and Generalization

A model should be judged by how well it generalizes to unseen data, not by how well it fits the training set.

## Train, validation, test

- **train set**: fit the parameters
- **validation set**: choose hyperparameters and compare models
- **test set**: final unbiased estimate of performance

Using the test set repeatedly for tuning leaks information and invalidates it as a final estimate.

## Bias-variance tradeoff

Expected prediction error can be decomposed conceptually as:

- **bias**: systematic error from an overly rigid model
- **variance**: sensitivity to sampling noise
- **irreducible noise**: randomness in the target

Typical patterns:

- high bias: underfitting
- high variance: overfitting

Examples:

- a shallow tree often has higher bias
- a deep unconstrained tree often has higher variance

## Cross-validation

When data is limited, $k$-fold cross-validation gives a more stable estimate.

Procedure:

1. split the data into $k$ folds
2. train on $k-1$ folds
3. validate on the held-out fold
4. repeat and average

This reduces dependence on one arbitrary split.

## Metrics

### Regression

- MSE / RMSE
- MAE
- $R^2$

### Classification

- accuracy
- precision, recall, F1
- ROC-AUC
- PR-AUC
- log loss / cross-entropy
- calibration error when probabilities matter

Metric choice should match the task cost structure.

Examples:

- fraud detection: recall or PR-AUC may matter more than accuracy
- credit scoring: calibration and ranking may both matter
- medical screening: false negatives may dominate the decision

## Confusion matrix

For binary classification:

- true positive
- false positive
- true negative
- false negative

From this:
$$
\text{precision}=\frac{TP}{TP+FP},\qquad
\text{recall}=\frac{TP}{TP+FN}.
$$

F1 score:
$$
F1 = 2\frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}.
$$

## Calibration vs discrimination

A model can rank examples well but be poorly calibrated.

- **discrimination**: can the model separate positives from negatives?
- **calibration**: do predicted probabilities match observed frequencies?

Example:

- if all predictions around 0.8 are truly positive about 80% of the time, the model is well calibrated

## Distribution shift

Generalization can fail when train and deployment distributions differ.

Examples:

- covariate shift
- label shift
- concept drift over time

This is why time-based validation or group-based validation is often necessary in real systems.

## Data leakage

Common leakage sources:

- target-derived features
- fitting preprocessing on the full dataset
- using future information in time series
- duplicates across train and test

Leakage can make a bad model look excellent.

## Small code example

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
print(scores.mean(), scores.std())
```

## What to remember

- Generalization, not training fit, is the real objective
- Match the metric to the operational cost
- Use the validation set for selection and the test set only once at the end
- Watch for leakage, imbalance, and distribution shift
