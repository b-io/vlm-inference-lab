# Documentation Map

This directory is organized by topic rather than by the historical source of each note. The goal is to make the
material easier to browse as a coherent reference set.

## Folder guide

- [`fundamentals/`](fundamentals/): general machine-learning concepts, losses, optimization, representations, model
  selection, decision trees as base learners, and ensemble methods as the umbrella theory for random forests and
  boosted trees.
- [`architectures/`](architectures/): neural-network families and architecture tradeoffs across CNNs, RNNs, LSTMs,
  GRUs, vision models, and related sequence models.
- [`transformers/`](transformers/): tokenization, decoding, masking, and positional encoding for Transformer-based
  models.
- [`multimodal/`](multimodal/): embeddings, VLM architecture families, document understanding, multilingual alignment,
  and referring-expression comprehension.
- [`serving/`](serving/): inference-time behavior such as batching, KV cache, latency/throughput tradeoffs, training
  vs. inference, and model-complexity implications for serving.
- [`systems/`](systems/): GPU execution, memory hierarchy, kernel bottlenecks, profiling, and platform portability.
- [`optimization/`](optimization/): model-compression and efficiency techniques such as distillation, pruning, and
  quantization.
- [`evaluation/`](evaluation/): task evaluation, metrics, and SLO-oriented quality assessment.
- [`tooling/`](tooling/): framework- and CLI-specific notes, such as vLLM and SGLang.
- [`deployment/`](deployment/): platform-specific deployment and orchestration walkthroughs.

## Suggested reading paths

### Path 1: Foundations and model selection

1. [Linear models and losses](fundamentals/linear_models_and_losses.md)
2. [Decision trees](fundamentals/decision_trees.md)
3. [Ensemble methods](fundamentals/ensemble_methods.md)
4. [Random forests](fundamentals/random_forests.md)
5. [Gradient-boosted trees](fundamentals/gradient_boosted_trees.md)
6. [Model evaluation and generalization](fundamentals/model_evaluation_and_generalization.md)
7. [Model selection and use cases](fundamentals/model_selection_and_use_cases.md)

This path is intentional: read [decision trees](fundamentals/decision_trees.md) as the building block, then
[ensemble methods](fundamentals/ensemble_methods.md) as the umbrella theory, then the specific tree ensembles
[random forests](fundamentals/random_forests.md) and [gradient-boosted trees](fundamentals/gradient_boosted_trees.md).

### Path 2: Foundations to architecture selection

1. [Optimization and backpropagation](fundamentals/optimization_and_backprop.md)
2. [Activation functions](fundamentals/activation_functions.md)
3. [Neural architecture tradeoffs](architectures/neural_architecture_tradeoffs.md)
4. [Sequential models](architectures/sequential_models.md)
5. [RNN, LSTM, GRU, and gradient stability](architectures/rnn_lstm_gru_and_gradient_stability.md)
6. [Vision models](architectures/vision_models.md)

### Path 3: Transformers and modern language-model internals

1. [Transformers, tokenization, and decoding](transformers/transformers_tokenization_and_decoding.md)
2. [Attention masking and attention patterns](transformers/attention_masking_and_attention_patterns.md)
3. [Position embeddings and positional encoding](transformers/position_embeddings_and_positional_encoding.md)
4. [Training vs. inference](serving/training_vs_inference.md)
5. [KV cache](serving/kv_cache.md)

### Path 4: Vision-language modeling

1. [Embeddings and CLIP-style alignment](multimodal/embeddings.md)
2. [VLM architectures and basics](multimodal/vlm_architectures_and_basics.md)
3. [Document understanding](multimodal/document_understanding.md)
4. [Referring expression comprehension](multimodal/referring_expression_comprehension.md)
5. [Multilingual alignment](multimodal/multilingual_alignment.md)
6. [VLM evaluation and SLOs](evaluation/vlm_evaluation_and_slos.md)

### Path 5: Serving and performance engineering

1. [Batching, latency, and throughput](serving/batching_latency_and_throughput.md)
2. [Advanced serving features](serving/advanced_serving_features.md)
3. [Model complexity, parallelism, and hardware](serving/model_complexity_parallelism_and_hardware.md)
4. [Profiling and optimization workflow](systems/profiling_and_optimization_workflow.md)
5. [Inference kernel bottlenecks](systems/inference_kernel_bottlenecks.md)
6. [GPU memory and kernel execution](systems/gpu_memory_and_kernel_execution.md)

### Path 6: Compression, tooling, and deployment

1. [Knowledge distillation](optimization/knowledge_distillation.md)
2. [Pruning and quantization](optimization/pruning_and_quantization.md)
3. [vLLM CLI and SGLang](tooling/vllm_cli_and_sglang.md)
4. [Runpod deployment guide](deployment/runpod_demo.md)
5. [Lambda deployment guide](deployment/lambda_demo.md)

## Complete document map

### Fundamentals

- [Activation functions](fundamentals/activation_functions.md)
- [Decision trees](fundamentals/decision_trees.md)
- [Ensemble methods](fundamentals/ensemble_methods.md) *(umbrella theory for random forests, boosting, stacking, and voting ensembles)*
- [Feature engineering and representations](fundamentals/feature_engineering_and_representations.md)
- [Gradient-boosted trees](fundamentals/gradient_boosted_trees.md)
- [Linear models and losses](fundamentals/linear_models_and_losses.md)
- [Model evaluation and generalization](fundamentals/model_evaluation_and_generalization.md)
- [Model selection and use cases](fundamentals/model_selection_and_use_cases.md)
- [Optimization and backpropagation](fundamentals/optimization_and_backprop.md)
- [Random forests](fundamentals/random_forests.md)

### Architectures

- [Neural architecture tradeoffs](architectures/neural_architecture_tradeoffs.md)
- [Sequential models](architectures/sequential_models.md)
- [RNN, LSTM, GRU, and gradient stability](architectures/rnn_lstm_gru_and_gradient_stability.md)
- [Vision models](architectures/vision_models.md)

### Transformers

- [Transformers, tokenization, and decoding](transformers/transformers_tokenization_and_decoding.md)
- [Attention masking and attention patterns](transformers/attention_masking_and_attention_patterns.md)
- [Position embeddings and positional encoding](transformers/position_embeddings_and_positional_encoding.md)

### Multimodal

- [Embeddings and CLIP-style alignment](multimodal/embeddings.md)
- [VLM architectures and basics](multimodal/vlm_architectures_and_basics.md)
- [Document understanding](multimodal/document_understanding.md)
- [Multilingual alignment](multimodal/multilingual_alignment.md)
- [Referring expression comprehension](multimodal/referring_expression_comprehension.md)

### Serving

- [Advanced serving features](serving/advanced_serving_features.md)
- [Batching, latency, and throughput](serving/batching_latency_and_throughput.md)
- [KV cache](serving/kv_cache.md)
- [Model complexity, parallelism, and hardware](serving/model_complexity_parallelism_and_hardware.md)
- [Training vs. inference](serving/training_vs_inference.md)

### Systems

- [CUDA vs. HIP / ROCm](systems/cuda_vs_hip.md)
- [GPU memory and kernel execution](systems/gpu_memory_and_kernel_execution.md)
- [Inference kernel bottlenecks](systems/inference_kernel_bottlenecks.md)
- [Profiling and optimization workflow](systems/profiling_and_optimization_workflow.md)

### Optimization

- [Knowledge distillation](optimization/knowledge_distillation.md)
- [Pruning and quantization](optimization/pruning_and_quantization.md)

### Evaluation

- [VLM evaluation and SLOs](evaluation/vlm_evaluation_and_slos.md)

### Tooling

- [vLLM CLI and SGLang](tooling/vllm_cli_and_sglang.md)

### Deployment

- [Lambda deployment guide](deployment/lambda_demo.md)
- [Runpod deployment guide](deployment/runpod_demo.md)
