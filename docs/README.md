# Documentation Map and Interview Study Guide

This directory mixes three kinds of material:
- **ML / architecture fundamentals**: how different model families work and when to use them.
- **Serving / systems notes**: batching, KV cache, benchmarking, GPU bottlenecks.
- **VLM-specific notes**: multimodal alignment, document understanding, REC, and evaluation.

## Recommended study order for this repo

### 1. Core neural-network architecture tradeoffs
- [Neural architecture tradeoffs](ml/neural_architecture_tradeoffs.md)
- [Sequential models](ml/sequential_models.md)
- [RNN, LSTM, GRU, and gradient stability](ml/rnn_lstm_gru_and_gradient_stability.md)
- [Vision models](ml/vision_models.md)
- [Vision-language models](ml/vision_language_models.md)
- [Model selection and use cases](ml/model_selection_and_use_cases.md)

### 2. Optimization and serving
- [Optimization and backpropagation](ml/optimization_and_backprop.md)
- [Knowledge distillation](optimization/knowledge_distillation.md)
- [Pruning and quantization](optimization/pruning_and_quantization.md)
- [KV cache](inference/kv_cache.md)
- [Advanced serving features](inference/advanced_serving_features.md)
- [Latency vs throughput](inference/latency_vs_throughput.md)
- [vLLM CLI and SGLang](tools/vllm_cli_and_sglang.md)

### 3. VLM-specific interview topics
- [VLM basics](vlm/vlm_basics.md)
- [Embeddings and CLIP-style alignment](vlm/embeddings.md)
- [Document understanding](vlm/document_understanding.md)
- [Referring expression comprehension](vlm/referring_expression_comprehension.md)
- [Multilingual alignment](vlm/multilingual_alignment.md)
- [VLM evaluation and SLOs](evaluation/vlm_evaluation_and_slos.md)

### 4. GPU / kernel / hardware reasoning
- [Inference kernel bottlenecks](cuda/inference_kernel_bottlenecks.md)
- [CUDA memory model](cuda/cuda_memory_model.md)
- [Shared vs global memory](cuda/shared_vs_global_memory.md)
- [CUDA vs HIP / ROCm](cuda/cuda_vs_hip.md)

## Coverage notes

The repo now covers the major architecture families you may need to discuss in interview:
- **MLP / residual MLP**
- **CNN / ResNet / U-Net / FPN / ViT / Swin**
- **RNN / LSTM / GRU / temporal CNN / Transformer**
- **Autoencoders / VAEs / diffusion backbones / graph neural networks / mixture-of-experts**
- **Dual-encoder, fusion, projector-LLM, and document-oriented VLM families**

## Best interview framing

For this role, do not just memorize model names. Be able to answer four questions for each family:

1. **What problem is the architecture trying to solve?**
2. **What inductive bias does it encode?**
3. **What are the optimization and serving consequences?**
4. **When would you prefer it over the alternatives?**

That framing is stronger than a taxonomy-only answer.
