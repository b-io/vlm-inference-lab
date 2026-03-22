# Decision Trees

A decision tree is a recursive partitioning algorithm that splits the input space into regions and predicts a constant label or value in each region.

The model represents a piecewise-constant function:
- **classification**: predict a class or class distribution in each leaf
- **regression**: predict an average value in each leaf

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

This is why ensembles such as Random Forests and Gradient Boosted Trees are so effective.

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
- **post-pruning**: grow a larger tree, then prune subtrees whose complexity is not justified by validation performance or a regularized objective

Cost-complexity pruning adds a penalty on the number of leaves.

## What to remember

- Trees solve local impurity reduction problems by greedy splitting
- They are flexible and interpretable, but unstable and prone to overfitting
- They model interactions automatically, but only through hierarchical axis-aligned partitions
- Ensembles reduce the variance and usually improve accuracy substantially
