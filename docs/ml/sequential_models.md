# Sequential Models

Sequential models process data with an explicit ordering structure:
$$
(x_1, x_2, \dots, x_T).
$$
The central problem is to model dependencies across positions, either for prediction, representation learning, or conditional generation.

## Problem formulation

Typical objectives are:
- **Sequence classification**: predict $y$ from an ordered input sequence
- **Sequence labeling**: predict $y_t$ for each position $t$
- **Language modeling**: estimate
  $$
  p(x_1,\dots,x_T)=\prod_{t=1}^T p(x_t\mid x_{<t})
  $$
- **Sequence-to-sequence modeling**: map one sequence to another, e.g. translation

The main design question is how information from earlier and later positions influences the current representation.

## RNNs

A recurrent neural network maintains a hidden state updated step by step:
$$
h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b).
$$

Interpretation:
- $h_t$ is a summary of the prefix $(x_1,\dots,x_t)$
- the same parameters are reused across time
- order is handled implicitly by the recurrent update

### What it tries to solve

An RNN compresses past context into a fixed-size state and updates it online. This is natural when:
- inputs arrive sequentially
- streaming is required
- memory and latency per step must stay small

### Why recurrence helps

- natural handling of variable-length sequences
- parameter sharing across time
- online inference: process one token/frame at a time

### Main limitation

Long-range information must pass through many recurrent steps. The gradient becomes a product of many Jacobians, which can shrink or grow exponentially. This is the vanishing/exploding gradient problem.

## LSTM and GRU

LSTM introduces a gated memory cell:
$$
i_t = \sigma(\cdot), \quad f_t = \sigma(\cdot), \quad o_t = \sigma(\cdot), \quad g_t = \tanh(\cdot)
$$
$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t).
$$

### What LSTM tries to solve

It reduces vanishing gradients by creating an additive memory path through $c_t$. The forget gate can preserve information over long spans better than a plain RNN.

### Why the design looks this way

- **Sigmoid gates** produce values in $[0,1]$, which naturally express keep/write/expose decisions
- **Tanh content** is signed and bounded, so memory updates can add or subtract information
- the cell state creates a controlled linear path through time

### Tradeoffs

- better long-range memory than vanilla RNNs
- still sequential in time, so parallelism is limited
- more parameters and more expensive per step than a plain RNN

GRUs use fewer gates and a simpler state update. They often perform competitively with fewer parameters, but give less explicit memory-state separation than LSTMs.

## Attention over recurrent models

Before Transformers, sequence-to-sequence models often used an encoder RNN plus an attention mechanism over encoder states. The decoder can then select relevant source positions instead of relying on a single fixed bottleneck state.

This solves the problem that a single hidden vector often compresses long inputs poorly.

## Transformers

A Transformer replaces recurrence with self-attention. For token representations $X$, form:
$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V,
$$
and compute
$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

### What self-attention tries to solve

Instead of forcing information through a recurrent chain, each position can directly interact with all other positions in the layer.

### Why it helps

- short path length between distant positions
- highly parallelizable across sequence positions during training
- flexible dependency patterns: each token can attend to whichever positions matter most

### Main tradeoff

Full attention compares all pairs of positions.

For sequence length $n$ and hidden size $d$:
- self-attention time: roughly $O(n^2 d)$
- attention memory: roughly $O(n^2)$
- recurrent layer time: roughly $O(n d^2)$ with sequential dependence

So Transformers scale better with parallel hardware, but the quadratic cost in $n$ becomes expensive for long contexts.

## Main Transformer families

### Encoder-only

Use bidirectional self-attention over the full input.

Examples of tasks:
- classification
- token labeling
- retrieval embeddings

Why this architecture: all positions can use both left and right context, which is ideal for representation learning.

### Decoder-only

Use **causal masking**, so token $t$ can only attend to positions $< t$.

Objective:
$$
\max \sum_t \log p(x_t \mid x_{<t}).
$$

Why this architecture: it directly matches autoregressive generation.

### Encoder-decoder

The encoder builds a source representation; the decoder uses both causal self-attention and cross-attention to the source.

Why this architecture: it separates input understanding from conditional generation and is natural for translation, summarization, and structured transduction.

## Positional information

RNNs encode order through recurrence. Transformers must inject order explicitly via positional encodings or learned positional embeddings.

Without positional information, self-attention is permutation-equivariant and cannot distinguish different token orders.

## Practical comparison

| Model | Strength | Weakness | Best when |
|---|---|---|---|
| Vanilla RNN | Simple online sequence processing | Weak long-range memory | Small streaming problems, teaching intuition |
| LSTM / GRU | Better memory through gating | Still sequential, harder to parallelize | Time series, speech, moderate-length sequences, online settings |
| Transformer encoder | Strong contextual representations | Quadratic attention cost | Sequence understanding, embeddings, classification |
| Transformer decoder | Excellent autoregressive modeling | Expensive for long contexts | Language modeling and generation |
| Encoder-decoder Transformer | Strong conditional generation | More complex than single-stack models | Translation, summarization, seq2seq tasks |

## Minimal code sketches

### RNN/LSTM-style sequence model

```python
h = h0
for x_t in sequence:
    h = cell(x_t, h)
logits = classifier(h)
```

### Transformer-style causal language model

```python
x = token_embeddings(tokens) + positional_embeddings(positions)
for block in transformer_blocks:
    x = block(x, causal_mask=True)
logits = lm_head(x)
```

## What to remember

- Sequential modeling is about representing dependencies across ordered positions
- RNNs use recurrence and a hidden state; LSTMs/GRUs stabilize memory with gates
- Transformers use self-attention to expose direct token-token interactions
- The main systems tradeoff is **sequential recurrence vs quadratic all-to-all attention**
