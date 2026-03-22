#!/bin/bash

# Default values
MODEL=${1:-"facebook/opt-125m"}
PORT=8000

echo "--------------------------------------------------------"
echo "Starting vLLM with model: $MODEL on port: $PORT"
echo "--------------------------------------------------------"

docker run -d --name vllm-server \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p $PORT:8000 \
    --env HF_TOKEN=$HF_TOKEN \
    vllm/vllm-openai:latest \
    --model $MODEL
