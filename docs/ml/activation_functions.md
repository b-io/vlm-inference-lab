# Activation Functions — Expert Interview Guide

This note is designed for technical interviews where the audience already knows the basics and cares about **tradeoffs**, **architecture fit**, and **why a given activation is used in a specific setting**.

---

## 1) What an activation function fundamentally does

An activation function does more than “add nonlinearity.” In practice, it determines:

1. **Expressivity**
   - Without nonlinearities, a stack of linear or affine layers collapses to a single affine map.
   - Depth only becomes useful when nonlinearities are inserted between layers.

2. **Gradient flow**
   - The derivative of the activation changes how easily gradients propagate through many layers.
   - Saturating activations can make gradients vanish.
   - Piecewise-linear activations such as ReLU often improve optimization in deep feed-forward stacks.

3. **Representation bias**
   - Some activations encourage **sparse positive features** (ReLU).
   - Some encourage **signed, bounded states** (tanh).
   - Some naturally implement **soft gates** (sigmoid, GELU, SiLU).

4. **Semantics of the hidden state**
   - If a unit should represent a **gate or probability**, values in $[0,1]$ are natural.
   - If a unit should represent **signed content**, values in $[-1,1]$ are often more natural.

5. **Compute and implementation cost**
   - ReLU is extremely cheap.
   - GELU/SiLU are smoother but more expensive.
   - Gated FFN variants such as SwiGLU increase parameters and operations but often improve model quality.

---

## 2) The important axes, with examples

### A. Saturation
An activation **saturates** when for large positive or negative inputs its derivative becomes very small.

- **Sigmoid** saturates near 0 and 1.
- **tanh** saturates near $-1$ and $1$.
- **ReLU** does not saturate on the positive side, but is exactly flat on the negative side.

Why it matters:
- If the derivative is near zero, then backpropagated gradients shrink.
- This is one reason deep nets with sigmoid/tanh hidden layers were harder to train than ReLU-style networks.

### B. Zero-centering
A hidden activation is **zero-centered** if its output is distributed around 0.

- **tanh** is zero-centered.
- **sigmoid** is not; its range is $(0,1)$.
- **ReLU** is also not zero-centered because outputs are nonnegative.

Why it matters:
- Zero-centered hidden states often make optimization easier because the next layer sees inputs with positive and negative signs.
- In recurrent states, signed hidden content is often more natural than strictly positive content.

### C. Boundedness
An activation is **bounded** if its output stays inside a fixed interval.

- **sigmoid** is bounded in $(0,1)$.
- **tanh** is bounded in $(-1,1)$.
- **ReLU/GELU/SiLU** are not bounded above.

Why it matters:
- Bounded activations are useful when a quantity should behave like a gate or a controlled memory value.
- Unbounded activations often allow richer feature magnitudes but require the rest of the architecture to manage scale.

### D. Sparsity
Some activations produce exact zeros.

- **ReLU** outputs exactly 0 for negative inputs.
- **Leaky ReLU** reduces but does not eliminate this effect.
- **GELU/SiLU** are soft and generally do not produce exact zeros.

Why it matters:
- Sparsity can help representation learning and efficiency.
- In CNNs, sparse local feature activations often align well with feature-detection intuition.

### E. Smoothness
A smooth activation has a smooth derivative.

- **ReLU** has a kink at 0.
- **GELU** and **SiLU** are smooth.

Why it matters:
- Smooth activations often give more gradual, stable local changes.
- In large dense Transformer FFNs, smooth magnitude-dependent gating often works better than hard thresholding.

### F. Gating semantics
Some activations are best understood as gates.

- **sigmoid**: explicit multiplicative gate in $[0,1]$.
- **GELU**: smooth magnitude-dependent attenuation.
- **SiLU/Swish**: similar soft-gating interpretation.
- **SwiGLU/GEGLU**: one branch carries content and another branch gates it.

Why it matters:
- In RNNs/LSTMs, true gates are required to control memory flow.
- In Transformers, gated FFNs improve expressivity beyond a single one-branch nonlinearity.

---

## 3) Architecture-oriented comparison table

| Activation | Formula | Common architectures | Why it is used there | Main tradeoff |
|---|---|---|---|---|
| Sigmoid | $\sigma(x)=\frac{1}{1+e^{-x}}$ | LSTM/GRU gates, binary classifiers, logistic outputs | Best when a unit should behave like a gate or probability: output in $[0,1]$ naturally means “how much to keep / write / expose.” | Saturates in both tails, not zero-centered, poor generic hidden activation in deep nets |
| tanh | $\tanh(x)$ | RNN hidden states, LSTM candidate/state squashing, older CNNs (LeNet era) | Best when a hidden state should be signed, bounded, and zero-centered. In recurrent memory, content may need to increase or decrease the state, so $[-1,1]$ is more natural than $[0,1]$. | Saturates in both tails; slower gradient flow than ReLU-family activations in deep feed-forward stacks |
| ReLU | $\max(0,x)$ | Modern CNNs, generic MLPs, original Transformer FFN | Especially good in CNNs because it is cheap, promotes sparse local features, and avoids positive-side saturation, which helped deep vision models train much faster than tanh/sigmoid-style units. | Dead neurons on persistent negative inputs; discards all negative information; hard threshold |
| Leaky ReLU / PReLU | $\max(\alpha x, x)$ | CNNs and MLPs when dead ReLUs are a concern | Keeps most of ReLU’s benefits while preserving a small gradient on the negative side. Helpful when feature channels should not completely die. | Less sparse than ReLU; still piecewise linear and less smooth than GELU |
| GELU | $x\Phi(x)$ | BERT, many Transformer encoders, some vision/NLP models | Often preferred in Transformers because it is a smooth, magnitude-dependent gate. It preserves weak signals near zero better than ReLU and avoids hard thresholding in large dense FFN blocks. | More compute than ReLU; no exact sparsity |
| SiLU / Swish | $x\sigma(x)$ | Deep nets needing a smooth ReLU-like activation; also inside gated Transformer FFNs | Similar motivation to GELU: smooth nonlinearity, keeps small negative responses, often good optimization behavior. | More expensive than ReLU |
| GEGLU / SwiGLU | gated FFN variants | Modern LLM/Transformer FFN blocks | Better than plain one-branch activations when the architecture benefits from content $\times$ gate interactions. One projection carries content; another projection modulates it. This often improves Transformer quality. | More parameters/ops than a plain FFN |
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | Output layer for multiclass classification, next-token prediction | Used when outputs should form a probability distribution over classes/tokens. | Not a hidden-layer activation in the usual sense |

---

## 4) Why certain activations fit certain architectures

### 4.1 CNNs: why ReLU became dominant

CNNs learn many local feature detectors. A convolution channel often answers questions like:

- “Is there an edge here?”
- “Is there a corner-like pattern here?”
- “Is there a texture patch here?”

ReLU works very well here because:

1. **Feature detectors often want one-sided responses**
   - If a local pattern is present, a positive activation is useful.
   - If not, zero is a natural response.
   - This creates sparse feature maps.

2. **Positive-side non-saturation helps optimization**
   - Once a filter starts producing useful positive activations, gradients do not shrink the way they do with sigmoid/tanh.

3. **Cheap at scale**
   - CNNs apply activations over many spatial locations and channels, so cheap pointwise nonlinearities matter.

Why not tanh in modern CNNs?
- tanh gives signed bounded outputs, but deep CNNs benefited strongly from faster optimization with ReLU.
- In vision, exact negative activations are often less important than efficient sparse feature detection.

When use Leaky ReLU/PReLU instead?
- If many CNN channels die because their pre-activations stay negative, a small negative slope helps keep those channels trainable.

### 4.2 RNNs and LSTMs: why tanh and sigmoid coexist

In an LSTM:

$$
i_t=\sigma(\cdot),\quad f_t=\sigma(\cdot),\quad g_t=\tanh(\cdot),\quad o_t=\sigma(\cdot)
$$

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t,\quad h_t=o_t\odot\tanh(c_t)
$$

Interpretation:
- $f_t$: forget gate
- $i_t$: input gate
- $o_t$: output gate
- $g_t$: candidate content

Why **sigmoid** for the gates?
- A gate should mean “how much” to keep, write, or expose.
- Values in $[0,1]$ are exactly the right semantics.
- Negative values or values above 1 would turn a gate into an inversion or amplification mechanism instead of a simple mask.

Why **tanh** for the candidate content?
- The candidate is **content**, not a gate.
- Content should be able to be positive or negative.
- The update should be bounded and zero-centered.
- A one-sided sigmoid candidate would bias the recurrent state in a positive direction.

Why not ReLU for classical LSTM state content?
- ReLU is unbounded above and one-sided.
- For recurrent memory, that can make state dynamics less controlled.
- Classical LSTM design deliberately separates **bounded signed content** (tanh) from **gating control** (sigmoid).

### 4.3 Transformers: why GELU or SwiGLU instead of plain ReLU

Transformers use large dense feed-forward blocks at every layer. In these FFNs, ReLU works, but GELU and gated variants often work better.

Why **GELU** instead of ReLU?

1. **Smooth gating**
   - ReLU is a hard threshold at 0.
   - GELU attenuates small inputs smoothly instead of snapping them to zero.

2. **Weak-signal preservation**
   - In dense high-dimensional FFNs, many coordinates near zero may still carry useful information.
   - GELU preserves and scales these more gently.

3. **Optimization behavior**
   - The smooth derivative can make training dynamics less brittle than a hard threshold.

Why **SwiGLU/GEGLU** in modern LLMs?
- A plain FFN computes something like:
  $$
  \mathrm{FFN}(x)=W_2\,\phi(W_1x)
  $$
- A gated FFN computes something closer to:
  $$
  \mathrm{SwiGLU}(x)=W_o\big(\mathrm{SiLU}(W_gx) \odot W_vx\big)
  $$
- One branch holds content, another branch gates it.
- This gives a richer multiplicative interaction than a single activation on one branch.

This is especially valuable in LLMs because FFNs are a major part of model capacity.

---

## 5) Why GELU instead of ReLU? Crisp expert comparison

### ReLU
$$
\mathrm{ReLU}(x)=\max(0,x)
$$

Properties:
- piecewise linear
- sparse
- exact zeros on the negative side
- derivative is 1 on positive side, 0 on negative side

Best argument for it:
- very cheap
- very effective in CNNs and generic deep feed-forward nets
- avoids positive-side saturation

Main limitation:
- hard-threshold behavior may throw away weak but useful information
- dead neurons if a unit stays negative

### GELU
$$
\mathrm{GELU}(x)=x\Phi(x)
$$

Interpretation:
- input is scaled by a smooth probability-like gating term
- large positive values pass mostly unchanged
- large negative values are suppressed
- values near zero are attenuated smoothly, not clipped hard

Best argument for it:
- better suited to dense Transformer FFNs where small activations can still contain useful signal
- smoother than ReLU
- often empirically stronger in Transformer families

Main limitation:
- more expensive than ReLU
- does not give exact sparsity

A concise expert sentence:

> ReLU is a hard sign gate that is excellent for sparse feature extraction and cheap optimization, especially in CNNs. GELU is a smooth magnitude-dependent gate that tends to work better in dense Transformer FFNs where preserving weak signals near zero is useful.

---

## 6) Why tanh rather than sigmoid for recurrent hidden content?

This distinction is extremely important.

### Sigmoid as gate
- Range: $(0,1)$
- Meaning: “how much”
- Good for multiplicative masking

### tanh as content
- Range: $(-1,1)$
- Meaning: signed content
- Good when state must both increase and decrease

Example intuition:
- A memory cell may want to encode “positive evidence” and “negative evidence.”
- With tanh, the candidate content can push the state in either direction.
- With sigmoid, the candidate content would always be nonnegative, which is much less expressive.

So the clean rule is:

- **sigmoid** for control
- **tanh** for content

---

## 7) Softmax: why it belongs at the output, not as a generic hidden activation

Softmax is different from the others.

For logits $z_1, \dots, z_K$:

$$
\mathrm{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Why it is used:
- produces a normalized categorical distribution
- outputs sum to 1
- ideal for multiclass classification and next-token prediction

Why not use it as a normal hidden activation?
- It couples all units together.
- Its job is probabilistic normalization, not local feature transformation.
- It is usually the **final readout**, not an internal representation nonlinearity.

---

## 8) Quick decision rules

### Use ReLU when:
- you are in a CNN or generic MLP
- you want cheap, strong baseline behavior
- sparse positive activations are desirable

### Use Leaky ReLU / PReLU when:
- you like ReLU behavior but worry about dead channels/neurons

### Use tanh when:
- you need signed bounded hidden content
- you are discussing classical recurrent state design

### Use sigmoid when:
- the unit is a gate or binary probability
- you need semantics in $[0,1]$

### Use GELU when:
- you are in Transformer encoders or dense FFNs
- smooth gating is more appropriate than hard thresholding

### Use SwiGLU / GEGLU when:
- discussing modern LLM FFNs
- you want the expert answer for current Transformer practice

### Use softmax when:
- the output must be a categorical distribution over classes or tokens

---

## 9) Interview-ready summary

> Activation functions are not only about adding nonlinearity. They determine gradient flow, saturation behavior, sparsity, centering, boundedness, and whether a unit acts like content or a gate. ReLU became dominant in CNNs because it is cheap, sparse, and avoids positive-side saturation. Sigmoid remains essential for gates because $[0,1]$ has the right semantics for “how much.” tanh is preferred for recurrent content because it is signed, bounded, and zero-centered. In Transformers, GELU and gated variants such as SwiGLU are often preferred because dense FFN blocks benefit from smooth or multiplicative gating rather than hard thresholding.

---

## 10) One-line expert contrasts

- **Sigmoid vs tanh:** sigmoid is for gates; tanh is for signed content.
- **ReLU vs GELU:** ReLU is sparse and cheap; GELU is smooth and better matched to dense Transformer FFNs.
- **ReLU vs Leaky ReLU:** Leaky ReLU sacrifices some sparsity to avoid dead units.
- **GELU vs SwiGLU:** GELU is a one-branch smooth activation; SwiGLU adds explicit multiplicative gating and often improves Transformer quality.
- **Softmax vs sigmoid:** softmax normalizes across classes; sigmoid treats outputs independently.
