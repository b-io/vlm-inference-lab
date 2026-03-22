# Linear Models, Logistic Regression, and Loss Functions

Linear models are simple but fundamental: they formalize prediction as a weighted combination of features.

## Linear regression

Model:
$$
\hat y = w^\top x + b.
$$

Objective:
$$
L(w,b)=\frac{1}{n}\sum_{i=1}^n (\hat y_i-y_i)^2.
$$

Interpretation:
- coefficients tell how the prediction changes with each feature
- the model is linear in the parameters
- the learned surface is a hyperplane

### What problem it solves
Estimate a conditional mean $E[y\mid x]$ under a linear approximation.

### Tradeoffs
- easy to fit and interpret
- fast, stable, strong baseline
- limited for nonlinear structure unless features are engineered

## Logistic regression

For binary classification:
$$
z = w^\top x + b,\qquad
\hat p = \sigma(z)=\frac{1}{1+e^{-z}}.
$$

Decision rule:
$$
\hat y =
\begin{cases}
1 & \hat p \ge 0.5\\
0 & \hat p < 0.5
\end{cases}
$$

The model is linear in the **log-odds**:
$$
\log \frac{\hat p}{1-\hat p} = w^\top x + b.
$$

### Loss
Binary cross-entropy:
$$
L = -\frac{1}{n}\sum_{i=1}^n \left[y_i\log \hat p_i + (1-y_i)\log(1-\hat p_i)\right].
$$

### What problem it solves
Estimate a conditional Bernoulli probability $P(y=1\mid x)$ under a linear decision boundary in feature space.

### Tradeoffs
- probabilistic and interpretable
- strong calibration in many settings
- limited to linear boundaries unless features are expanded

## Regularization

### L2 / Ridge
$$
L_{\lambda} = L + \lambda \|w\|_2^2
$$
shrinks coefficients smoothly.

### L1 / Lasso
$$
L_{\lambda} = L + \lambda \|w\|_1
$$
encourages sparsity and feature selection.

Why regularize:
- reduce variance
- stabilize solutions under collinearity
- improve generalization

## Typical losses and when they are used

| Loss | Formula | Typical use | Why |
|---|---|---|---|
| MSE | $\frac{1}{n}\sum_i (\hat y_i-y_i)^2$ | Regression | Penalizes large errors strongly; convenient derivatives |
| MAE | $\frac{1}{n}\sum_i |\hat y_i-y_i|$ | Regression | More robust to outliers than MSE |
| Binary cross-entropy | $-\frac{1}{n}\sum_i [y_i\log p_i +(1-y_i)\log(1-p_i)]$ | Binary classification | Proper probabilistic objective for Bernoulli targets |
| Cross-entropy | $-\sum_k y_k\log p_k$ | Multiclass classification | Works with softmax probabilities |
| Hinge loss | $\max(0,1-yf(x))$ | SVM-style margin classifiers | Focuses on decision margin rather than probability fitting |

## Small code example

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=1000
)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]
```

## What to remember

- Linear regression predicts values; logistic regression predicts probabilities
- Both are linear in the features unless features are transformed
- Regularization is often as important as the base model
- These models are simple, interpretable, and very strong baselines
