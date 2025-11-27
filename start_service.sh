#!/bin/bash

# MiniMax M2 Service Startup Script
# This script starts the vLLM server for MiniMax-M2 model

# Exit on any error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting MiniMax M2 Service...${NC}"

# Set environment variables for fast GPU loading
export SAFETENSORS_FAST_GPU=1

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate virtual environment
source /home/naresh/venvs/minimax-m2-service/bin/activate

echo -e "${YELLOW}Virtual environment activated${NC}"

# Model configuration
MODEL_PATH="/home/naresh/models/minimax-m2"
MODEL_NAME="MiniMaxAI/MiniMax-M2"
PORT=7104
HOST="0.0.0.0"
DTYPE="bfloat16"
MAX_MODEL_LEN=128000  # Full 128K context support
TENSOR_PARALLEL_SIZE=4  # Using 4x H100 GPUs
GPU_MEMORY_UTIL=0.95
MAX_NUM_SEQS=16

# Default generation parameters (caller can override)
TEMPERATURE=1.0
TOP_P=0.95
TOP_K=20
MAX_TOKENS=16384

# MiniMax M2 specific flags
ENABLE_AUTO_TOOL_CHOICE="--enable-auto-tool-choice"
TOOL_CALL_PARSER="--tool-call-parser minimax_m2"
REASONING_PARSER="--reasoning-parser minimax_m2"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. CUDA may not be installed correctly.${NC}"
    exit 1
fi

# Check GPU availability
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo -e "${GREEN}Detected $GPU_COUNT GPU(s)${NC}"

if [ "$GPU_COUNT" -lt "$TENSOR_PARALLEL_SIZE" ]; then
    echo -e "${RED}Error: Requested $TENSOR_PARALLEL_SIZE GPUs but only $GPU_COUNT available${NC}"
    exit 1
fi

# Display GPU information
echo -e "${GREEN}GPU Information:${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -n $TENSOR_PARALLEL_SIZE

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}Error: Port $PORT is already in use${NC}"
    echo "Please stop the existing service or choose a different port"
    exit 1
fi

# Check if model exists locally
if [ -d "$MODEL_PATH" ] && [ "$(ls -A $MODEL_PATH)" ]; then
    echo -e "${GREEN}Using local model from: $MODEL_PATH${NC}"
    MODEL_ARG="$MODEL_PATH"
else
    echo -e "${YELLOW}Model not found locally. Will download from HuggingFace: $MODEL_NAME${NC}"
    echo -e "${YELLOW}This may take some time (~220GB download)...${NC}"
    MODEL_ARG="$MODEL_NAME"
    # Create model directory if it doesn't exist
    mkdir -p "$MODEL_PATH"
fi

# Create logs directory if it doesn't exist
mkdir -p /home/naresh/minimax-m2-service/logs

echo ""
echo -e "${GREEN}Starting vLLM server with following configuration:${NC}"
echo "  Model: $MODEL_ARG"
echo "  Port: $PORT"
echo "  Host: $HOST"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE GPUs"
echo "  Max Model Length: $MAX_MODEL_LEN tokens (128K context)"
echo "  GPU Memory Utilization: ${GPU_MEMORY_UTIL}"
echo "  Max Concurrent Sequences: $MAX_NUM_SEQS"
echo "  Data Type: $DTYPE"
echo ""
echo "  Default Generation Parameters:"
echo "    Temperature: $TEMPERATURE"
echo "    Top-P: $TOP_P"
echo "    Top-K: $TOP_K"
echo "    Max Tokens: $MAX_TOKENS"
echo ""
echo "  MiniMax M2 Features:"
echo "    - Auto Tool Choice: Enabled"
echo "    - Tool Call Parser: minimax_m2"
echo "    - Reasoning Parser: minimax_m2 (strips <think> tags)"
echo ""
echo -e "${YELLOW}Note: Service will be accessible at http://localhost:$PORT${NC}"
echo -e "${YELLOW}Logs will be saved to: /home/naresh/minimax-m2-service/logs/service.log${NC}"
echo -e "${YELLOW}Model will use ~220GB GPU memory across 4 GPUs${NC}"
echo ""
echo -e "${GREEN}Starting server... (this may take a few minutes for initial model loading)${NC}"
echo ""

# Start vLLM server with logging (output to both console and log file)
vllm serve "$MODEL_ARG" \
  --port "$PORT" \
  --host "$HOST" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --trust-remote-code \
  --max-num-seqs "$MAX_NUM_SEQS" \
  $ENABLE_AUTO_TOOL_CHOICE \
  $TOOL_CALL_PARSER \
  $REASONING_PARSER \
  2>&1 | tee /home/naresh/minimax-m2-service/logs/service.log

