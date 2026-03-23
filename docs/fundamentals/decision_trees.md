# Decision Trees

A decision tree is a recursive partitioning algorithm that splits the input space into regions and predicts a constant
label or value in each leaf. It is a **single model**, not an ensemble method, but it is one of the most important base
learners used inside tree ensembles.

The model represents a piecewise-constant function:

- **classification**: predict a class or class distribution in each leaf
- **regression**: predict an average value in each leaf

Related notes:

- [Ensemble methods](ensemble_methods.md)
- [Random forests](random_forests.md)
- [Gradient-boosted trees](gradient_boosted_trees.md)
- [Model selection and use cases](model_selection_and_use_cases.md)

## What problem the algorithm is solving

At each node, the algorithm searches for a split
$$
x_j \le t
$$
that most improves node purity.

The greedy optimization problem is:

1. choose a feature $j$
2. choose a threshold $t$
3. partition the data into left/right children
4. maximize impurity reduction

The algorithm is greedy, not globally optimal.

## Classification criteria

### Entropy

If a node contains class proportions $p_1,\dots,p_K$, the entropy is
$$
H(p) = -\sum_{k=1}^K p_k \log p_k.
$$

### Gini impurity

$$
G(p) = 1 - \sum_{k=1}^K p_k^2.
$$

Both are low when the node is pure and high when classes are mixed.

The split is chosen to maximize information gain / impurity decrease:
$$
\Delta I = I(\text{parent}) - \frac{n_L}{n} I(\text{left}) - \frac{n_R}{n} I(\text{right}).
$$

## Regression criterion

For regression, a standard choice is variance / squared-error reduction.
If a leaf predicts the mean $\bar y$, then its squared loss is
$$
\sum_i (y_i - \bar y)^2.
$$
A good split is one that reduces this loss the most.

## Exact algorithm

For each node:

1. loop over candidate features
2. sort or scan candidate thresholds
3. evaluate the impurity reduction
4. take the best split
5. recurse until a stopping rule is met

Typical stopping rules:

- maximum depth
- minimum samples per split
- minimum samples per leaf
- zero impurity decrease
- pure node

## Complexity intuition

If there are $n$ samples and $d$ features:

- a naive split search is expensive because many thresholds must be tested
- in practice, implementations sort features or use efficient scans
- training cost grows with both data size and number of features
- prediction is cheap: just follow one root-to-leaf path

Inference complexity is roughly proportional to tree depth.

## Why trees are useful

### Advantages

- naturally handle nonlinear decision boundaries
- can model feature interactions without manual feature engineering
- require little scaling/normalization
- interpretable compared with deep models
- support mixed feature types reasonably well

### Tradeoffs

- high variance; small data changes can produce a different tree
- axis-aligned splits may be inefficient for oblique boundaries
- piecewise-constant predictions can be crude
- greedy training is not globally optimal
- unconstrained trees overfit easily

## From a single tree to tree ensembles

A single tree is often best viewed as the building block for stronger ensemble methods.

### Why one deep tree can overfit

Deep trees have low bias but high variance.
They can react strongly to small perturbations in the data, especially when many splits are available.

### Why bagging helps

If we average many trees whose errors are not perfectly correlated, the variance drops.
A useful approximation for the variance of the average of $B$ trees is

$$
\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2,
$$

where $\sigma^2$ is the variance of an individual tree and $\rho$ is the pairwise correlation.

This shows the two main levers:

- more trees reduce the independent part of the variance
- less correlation makes averaging much more effective

### How this leads to random forests

A [random forest](random_forests.md) uses:

1. **bagging**: bootstrap samples of the data
2. **feature subsampling**: random candidate features at each split
3. **aggregation**: averaging or majority voting across trees

This improves generalization mainly by reducing variance.

### How boosting is different

[Gradient-boosted trees](gradient_boosted_trees.md) do **not** train trees independently.
They train trees sequentially so that each new tree corrects the current ensemble.

That makes boosting more about stage-wise error correction and bias reduction, while bagging is more directly about
variance reduction.

## Example

Suppose the target is "approve loan" and features are:

- income
- debt ratio
- savings

A tree may learn:

1. if debt ratio $> 0.45$, reject
2. else if income $> 80{,}000$, approve
3. else if savings $> 20{,}000$, approve
4. else reject

This is easy to read, but the resulting surface is a set of axis-aligned rectangles.

## Small code example

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=10,
    random_state=0
)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
```

## Pruning

A fully grown tree often overfits. Two common strategies:

- **pre-pruning**: stop growing early via depth/leaf constraints
- **post-pruning**: grow a larger tree, then prune subtrees whose complexity is not justified by validation performance
  or a regularized objective

Cost-complexity pruning adds a penalty on the number of leaves.

## What to remember

- Trees solve local impurity reduction problems by greedy splitting.
- They are flexible and interpretable, but unstable and prone to overfitting.
- They model interactions automatically, but only through hierarchical axis-aligned partitions.
- Random forests improve generalization mainly by bagging and variance reduction.
- Gradient boosting builds a stronger predictor stage by stage from many small trees.
