# Random Forests

A random forest is an ensemble of decision trees trained with **bagging** and **feature subsampling**.
For prediction, it combines the trees by:

- **classification**: majority vote or averaged class probabilities
- **regression**: arithmetic average

## What problem random forests are solving

A single deep decision tree has low bias but often **high variance**:
small changes in the training data can produce a very different tree.

Random forests keep the expressive power of trees while improving generalization by averaging many of them.

The two key ideas are:

1. **bagging**: train each tree on a bootstrap sample
2. **random subspace selection**: at each split, search only a random subset of features

The first creates multiple versions of the training set.
The second decorrelates the trees further.

## Why bagging helps generalization

If each tree has variance $\sigma^2$ and pairwise correlation $\rho$, the average of $B$ trees has approximate variance

$$
\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2.
$$

This explains why random forests generalize better than one tree:

- averaging reduces the independent part of the noise
- feature subsampling reduces correlation between trees
- lower correlation means averaging is more effective

So bagging is mainly a **variance-reduction** method.

## The algorithm

Let the training set be $\{(x_i, y_i)\}_{i=1}^n$.
To train a forest with $B$ trees:

1. For each tree $b = 1, \dots, B$:
    1. draw a **bootstrap sample** of size $n$ from the training set, with replacement
    2. grow a decision tree on that sample
    3. at each internal node:
        - sample $m_{\text{try}}$ features uniformly at random from the full feature set
        - among only those features, choose the split with the largest impurity decrease
    4. continue until the stopping rule is met, often with deep or fully grown trees
2. Aggregate predictions across all trees.

Prediction:

- regression:
  $$
  \hat y(x) = \frac{1}{B}\sum_{b=1}^B T_b(x)
  $$
- classification:
  $$
  \hat y(x) = \mathrm{mode}\{T_1(x), \dots, T_B(x)\}
  $$

Here $T_b(x)$ is the prediction of tree $b$.

## Why feature subsampling matters

Suppose one feature is extremely strong.
If every tree sees all features at every split, many trees will start with the same root split and become highly
correlated.

Randomly restricting the candidate features at each split forces some trees to explore alternative structures.
That typically makes the forest more diverse, which improves the variance reduction from averaging.

## Out-of-bag (OOB) estimation

A bootstrap sample leaves out about one third of the original examples on average.
Those omitted examples are called **out-of-bag** for that tree.

This gives a built-in validation idea:

1. for each training example, collect predictions only from trees where that example was out-of-bag
2. compare those predictions with the true label/value
3. compute an OOB estimate of generalization performance

OOB error is often a very useful internal estimate, especially when data is not abundant.

## Important hyperparameters

### Number of trees (`n_estimators`)

More trees usually reduce variance and stabilize the estimate, at the cost of more memory and inference time.
Performance often plateaus after some point.

### Features per split (`max_features` or $m_{\text{try}}$)

- smaller values: more decorrelation, potentially lower variance, but weaker individual trees
- larger values: stronger individual trees, but more correlation

### Tree depth / leaf constraints

Forests often use fairly deep trees, because averaging controls variance.
Still, limits such as `max_depth`, `min_samples_leaf`, and `min_samples_split` can help when data is noisy.

### Bootstrap on/off

Standard random forests use bootstrap sampling.
Turning it off changes the method and often reduces the classic bagging effect.

## Strengths

- strong baseline for tabular data
- little preprocessing needed
- handles nonlinear interactions automatically
- robust to monotonic feature scaling
- naturally supports feature-importance style analysis
- OOB validation is convenient

## Limitations

- less interpretable than a single tree
- larger memory footprint than one tree
- prediction can be slower because many trees are evaluated
- piecewise-constant structure can be less sample-efficient than boosted trees on some tabular problems
- extrapolation in regression is usually weak

## Random forests vs gradient boosting

A practical contrast:

- **random forest**: many trees trained independently, then averaged
- **gradient boosting**: trees trained sequentially to correct the current ensemble

Random forests are often easier to tune and more robust by default.
Gradient-boosted trees are often stronger when carefully tuned on structured tabular data.

See also: [Gradient-boosted trees](gradient_boosted_trees.md).

## Small code example

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=5, oob_score=True, random_state=0,
        n_jobs=-1, )
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
print(clf.oob_score_)
```

## What to remember

- Random forests are bagged decision trees with feature subsampling.
- Bagging helps generalization mainly by reducing variance.
- Feature subsampling reduces correlation, which makes averaging more effective.
- Random forests are a strong, robust default for tabular data.
- They usually trade some interpretability and efficiency for better generalization.
