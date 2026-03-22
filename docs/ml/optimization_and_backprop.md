# Optimization, Derivatives, and Backpropagation

Training a neural network means minimizing a loss function with respect to model parameters.

## Derivative: the local rate of change

For a scalar function $f(x)$,
$$
f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}.
$$

Interpretation:
- the slope of the tangent line
- the best local linear approximation
- sensitivity of the output to a small change in the input

Locally,
$$
f(x+h)\approx f(x)+f'(x)h.
$$

## Gradient

For a multivariable function $L(\theta_1,\dots,\theta_p)$, the gradient is
$$
\nabla_\theta L =
\begin{bmatrix}
\frac{\partial L}{\partial \theta_1}\\
\vdots\\
\frac{\partial L}{\partial \theta_p}
\end{bmatrix}.
$$

It points in the direction of steepest increase of the loss. Gradient descent therefore moves in the negative gradient direction.

## Forward vs backward propagation

### Forward pass
Given parameters $\theta$, compute:
1. pre-activations
2. activations
3. predictions
4. loss

### Backward pass
Use the chain rule to compute gradients of the loss with respect to every parameter.

If
$$
L = f(g(h(x))),
$$
then
$$
\frac{dL}{dx} = \frac{dL}{df}\frac{df}{dg}\frac{dg}{dh}\frac{dh}{dx}.
$$

Backpropagation is just systematic application of this rule to a computational graph.

## Example: 2-layer network

Take
$$
z_1 = w_1x + b_1,\qquad h=\mathrm{ReLU}(z_1)
$$
$$
z_2 = w_2h+b_2,\qquad \hat y=\sigma(z_2)
$$
with binary cross-entropy loss
$$
L = -\left[y\log \hat y + (1-y)\log(1-\hat y)\right].
$$

A useful identity is:
$$
\frac{\partial L}{\partial z_2} = \hat y - y
$$
for sigmoid output with binary cross-entropy.

Then
$$
\frac{\partial L}{\partial w_2} = (\hat y-y)h,\qquad
\frac{\partial L}{\partial b_2} = \hat y-y.
$$

The gradient passed to the hidden unit is
$$
\frac{\partial L}{\partial h} = (\hat y-y)w_2.
$$

Back through ReLU:
$$
\frac{\partial h}{\partial z_1} =
\begin{cases}
1 & z_1>0\\
0 & z_1<0
\end{cases}
$$

Hence
$$
\frac{\partial L}{\partial z_1} =
\frac{\partial L}{\partial h}\frac{\partial h}{\partial z_1}.
$$

Finally,
$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z_1}x,\qquad
\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1}.
$$

## Parameter update

Gradient descent uses
$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$
where $\eta$ is the learning rate.

Mini-batch SGD uses a noisy estimate of the full gradient.

## Optimizer families

### SGD
Simple and robust:
$$
\theta_{t+1} = \theta_t - \eta g_t.
$$

### Momentum
Adds an exponentially decaying velocity:
$$
v_{t+1} = \beta v_t + g_t,\qquad
\theta_{t+1} = \theta_t - \eta v_{t+1}.
$$

Useful when the surface has narrow valleys or oscillatory directions.

### Adam
Tracks first and second moments of the gradient:
$$
m_t=\beta_1 m_{t-1} + (1-\beta_1)g_t,\qquad
v_t=\beta_2 v_{t-1} + (1-\beta_2)g_t^2.
$$

Adam is often faster to tune, though SGD with momentum sometimes generalizes better.

## Why gradients can vanish or explode

When many Jacobians are multiplied together:
- derivatives smaller than 1 can shrink gradients toward 0
- derivatives larger than 1 can blow them up

This is why initialization, normalization, residual connections, and activation choice matter.

## Small code example

```python
optimizer.zero_grad()
logits = model(x_batch)
loss = criterion(logits, y_batch)
loss.backward()      # backpropagation
optimizer.step()     # parameter update
```

## What to remember

- Forward pass computes predictions and loss
- Backward pass computes gradients by the chain rule
- Optimization updates parameters to reduce the loss
- Training stability depends on both the objective and the geometry of gradient flow
