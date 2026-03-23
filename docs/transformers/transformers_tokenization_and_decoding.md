# Transformers, Tokenization, and Decoding

A Transformer is a sequence model built around self-attention rather than recurrence.

## Core attention mechanism

Given token representations, form queries $Q$, keys $K$, and values $V$ and compute
$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Interpretation:

- $QK^\top$ measures pairwise compatibility
- softmax turns scores into attention weights
- the weighted sum of $V$ aggregates information from other positions

## Why self-attention matters

Every token can directly interact with every other token in the same layer.

Compared with an RNN:

- shorter path length between distant positions
- much more parallelism during training
- quadratic memory/time in sequence length for full attention

For sequence length $n$ and hidden size $d$:

- self-attention: roughly $O(n^2d)$ time and $O(n^2)$ attention memory
- recurrent layer: roughly $O(nd^2)$ time with sequential dependence

## Positional information

Transformers also need an explicit notion of order because self-attention does not automatically encode token position.
In practice, this is handled with positional embeddings or attention-level schemes such as sinusoidal encoding, relative
bias, RoPE, or ALiBi.

For a deeper treatment,
see [Position embeddings and positional encoding](position_embeddings_and_positional_encoding.md).

Masks are the other half of the story: they define which token-to-token interactions are legal in encoder-only,
decoder-only, encoder-decoder, and multimodal systems.
See [Attention masking and attention patterns](attention_masking_and_attention_patterns.md).

## Model families

- **Encoder-only** (e.g. BERT): bidirectional context, good for representation learning and understanding tasks
- **Decoder-only** (e.g. GPT-style): causal masking, good for generation
- **Encoder-decoder** (e.g. T5-style): conditional generation such as translation or summarization

## Tokenization

Modern LLMs typically use subword tokenization rather than whole-word tokenization.

Typical units:

- common whole words
- word pieces
- punctuation
- whitespace-aware chunks
- special tokens

Why subwords:

- avoid enormous whole-word vocabularies
- represent rare or unseen words as compositions of smaller pieces
- work better across code, names, numbers, and multilingual text

## Vocabulary and tokenizer internals

A tokenizer typically stores:

- token piece $\leftrightarrow$ token id maps
- merge rules or token scores
- normalization / pretokenization rules
- special tokens

BPE-style tokenizers often use merge-rank lookup tables. Trie-like prefix structures are useful for longest-prefix
matching or special token handling.

## Sampling and decoding

At generation step $t$, the model defines a probability mass function over the vocabulary:
$$
p(x_t \mid x_{\lt t}).
$$

### Greedy decoding

Choose the token with largest probability.

### Temperature

Rescale logits:
$$
p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}.
$$

- low $T$: sharper distribution, more deterministic
- high $T$: flatter distribution, more randomness

### Top-$p$ / nucleus sampling

Sort tokens by probability and keep the smallest set whose cumulative probability mass is at least $p$.

Interpretation:

- $p$ is a mass threshold, not a single-token probability
- after truncation, probabilities are renormalized before sampling

## Small code example

```python
probs = torch.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

## What to remember

- Attention lets each token mix information from other positions directly
- Transformer variants differ mainly by masking and encoder/decoder structure
- Tokenization is a preprocessing model of its own, usually subword-based
- Decoding quality depends strongly on the sampling strategy, not just the base model
