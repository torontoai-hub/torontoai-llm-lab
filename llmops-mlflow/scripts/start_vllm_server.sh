#!/bin/bash

# Default values
DEFAULT_PORT=8000
DEFAULT_MODEL="deepseek-coder-33b"
DEFAULT_TENSOR_PARALLEL=1
DEFAULT_GPU_MEMORY=40

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --tensor-parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --gpu-memory)
            GPU_MEMORY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set defaults if not provided
PORT=${PORT:-$DEFAULT_PORT}
MODEL=${MODEL:-$DEFAULT_MODEL}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-$DEFAULT_TENSOR_PARALLEL}
GPU_MEMORY=${GPU_MEMORY:-$DEFAULT_GPU_MEMORY}

echo "Starting vLLM server with:"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Tensor Parallel Degree: $TENSOR_PARALLEL"
echo "GPU Memory (GB): $GPU_MEMORY"

# Start the vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --gpu-memory-utilization $GPU_MEMORY \
    --trust-remote-code \
    --max-model-len 8192