# Activation Functions

Activation functions determine both the expressivity of a neural network and the optimization behavior of training.

A stack of affine maps without nonlinearities collapses to a single affine map:
$$
W_2(W_1x+b_1)+b_2 = (W_2W_1)x + (W_2b_1+b_2).
$$
Nonlinear activations are therefore what make depth useful. They also control gradient flow, saturation, centering, sparsity, and gating behavior.

## Important axes

### 1) Saturation
An activation **saturates** when its derivative becomes very small in part of its range.

Examples:
- Sigmoid: $\sigma'(x)=\sigma(x)(1-\sigma(x))$, which becomes small for large $|x|$
- Tanh: $\tanh'(x)=1-\tanh^2(x)$, which also vanishes in the tails

Consequence: gradients passed backward through many saturated layers can become tiny.

### 2) Zero-centering
An activation is **zero-centered** if its outputs are centered around 0.

Examples:
- Tanh outputs in $[-1,1]$ and is zero-centered
- Sigmoid outputs in $[0,1]$ and is not zero-centered

Why it matters: zero-centered activations often lead to better-conditioned optimization because updates are less systematically biased in one direction.

### 3) Boundedness
Some activations are bounded; others are unbounded.

- Sigmoid: bounded in $[0,1]$
- Tanh: bounded in $[-1,1]$
- ReLU: unbounded above

Bounded outputs are useful when the representation should have a controlled range, such as recurrent memory updates.

### 4) Sparsity
ReLU produces exact zeros on the negative side:
$$
\mathrm{ReLU}(x)=\max(0,x).
$$
This induces sparse activations, which can be useful in CNN feature maps and MLP hidden layers.

### 5) Smoothness
ReLU has a kink at 0. GELU and SiLU are smooth:
- GELU: $x\Phi(x)$
- SiLU: $x\sigma(x)$

Smooth activations often give gentler optimization behavior in large dense architectures such as Transformers.

### 6) Gating semantics
Some activations are naturally interpreted as gates.

- Sigmoid in $[0,1]$: "how much to pass"
- Tanh in $[-1,1]$: signed content
- SwiGLU/GEGLU: one branch carries content, another branch gates it

This is why LSTMs use sigmoid for gates and tanh for candidate content.

## Architecture-oriented comparison

| Activation | Formula | Common architectures | Why it is used there | Main tradeoff |
|---|---|---|---|---|
| Sigmoid | $\sigma(x)=\frac{1}{1+e^{-x}}$ | LSTM/GRU gates, binary classifiers, logistic outputs | Best when a unit should behave like a gate or probability: output in $[0,1]$ naturally means "how much to keep / write / expose." | Saturates in both tails, not zero-centered, poor generic hidden activation in deep nets |
| tanh | $\tanh(x)$ | RNN hidden states, LSTM candidate/state squashing, older CNNs (LeNet era) | Best when a hidden state should be signed, bounded, and zero-centered. In recurrent memory, content may need to increase or decrease the state, so $[-1,1]$ is more natural than $[0,1]$. | Saturates in both tails; slower gradient flow than ReLU-family activations in deep feed-forward stacks |
| ReLU | $\max(0,x)$ | Modern CNNs, generic MLPs, original Transformer FFN | Especially good in CNNs because it is cheap, promotes sparse local features, and avoids positive-side saturation, which helped deep vision models train much faster than tanh/sigmoid-style units. | Dead neurons on persistent negative inputs; discards all negative information; hard threshold |
| Leaky ReLU / PReLU | $\max(\alpha x, x)$ | CNNs and MLPs when dead ReLUs are a concern | Keeps most of ReLU's benefits while preserving a small gradient on the negative side. Helpful when feature channels should not completely die. | Less sparse than ReLU; still piecewise linear and less smooth than GELU |
| GELU | $x\Phi(x)$ | BERT, many Transformer encoders, some vision/NLP models | Often preferred in Transformers because it is a smooth, magnitude-dependent gate. It preserves weak signals near zero better than ReLU and avoids hard thresholding in large dense FFN blocks. | More compute than ReLU; no exact sparsity |
| SiLU / Swish | $x\sigma(x)$ | Deep nets needing a smooth ReLU-like activation; also inside gated Transformer FFNs | Similar motivation to GELU: smooth nonlinearity, keeps small negative responses, often good optimization behavior. | More expensive than ReLU |
| GEGLU / SwiGLU | gated FFN variants | Modern LLM/Transformer FFN blocks | Better than plain one-branch activations when the architecture benefits from content $\times$ gate interactions. One projection carries content; another projection modulates it. This often improves Transformer quality. | More parameters/ops than a plain FFN |
| Softmax | $\mathrm{softmax}(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}}$ | Output layer for multiclass classification, next-token prediction | Used when outputs should form a probability distribution over classes or tokens. | Not a hidden-layer activation in the usual sense |

## Why tanh in LSTMs and sigmoid for the gates?

A standard LSTM uses
$$
i_t=\sigma(\cdot), \quad f_t=\sigma(\cdot), \quad o_t=\sigma(\cdot), \quad g_t=\tanh(\cdot)
$$
and updates
$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t).
$$

Interpretation:
- $f_t, i_t, o_t \in [0,1]$ are multiplicative control signals
- $g_t \in [-1,1]$ is signed candidate content

Why this split is useful:
- a gate should suppress or pass, not invert the signal
- candidate content should be allowed to add or subtract information
- the hidden/readout state should remain bounded

## Tiny code example

```python
import torch
import torch.nn as nn

class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 128),
            nn.GELU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)
```

## What to remember

- Use **sigmoid** for gates or binary probabilities
- Use **tanh** when a state should be signed and bounded
- Use **ReLU / Leaky ReLU** for cheap, robust hidden layers, especially in CNNs and MLPs
- Use **GELU / SiLU** in large dense architectures where smooth optimization matters
- Use **softmax** only when the output should be a categorical distribution
