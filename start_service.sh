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

# Resolve the directory this script lives in — all paths are relative to it
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

echo -e "${GREEN}Starting MiniMax M2 Service...${NC}"

# Set CUDA environment — uses system symlink, always points to installed version
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Redirect HuggingFace downloads to the dedicated data disk
export HF_HOME=/mnt/hf-cache/huggingface
export HUGGINGFACE_HUB_CACHE=/mnt/hf-cache/huggingface/hub

# Ensure HF cache directories are writable by the current user
sudo mkdir -p "$HF_HOME/hub" "$HUGGINGFACE_HUB_CACHE"
sudo chown -R "$(id -u):$(id -g)" /mnt/hf-cache/huggingface

# Speed up SafeTensors model loading
export SAFETENSORS_FAST_GPU=1

# -----------------------------------------------------------------------
# Virtual environment — auto-create if it doesn't exist
# -----------------------------------------------------------------------
VENV_PATH="${HOME}/venvs/minimax-m2-service"

if [ ! -d "${VENV_PATH}" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating it...${NC}"
    mkdir -p "${HOME}/venvs"
    python3 -m venv "${VENV_PATH}"
    echo -e "${GREEN}Virtual environment created at: ${VENV_PATH}${NC}"

    source "${VENV_PATH}/bin/activate"
    echo -e "${YELLOW}Installing vLLM nightly (this may take several minutes)...${NC}"
    pip install --upgrade pip
    pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
    echo -e "${GREEN}vLLM installed${NC}"
else
    source "${VENV_PATH}/bin/activate"
    echo -e "${YELLOW}Virtual environment activated${NC}"
fi

# -----------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------
MODEL_PATH="/mnt/hf-cache/models/minimax-m2"
MODEL_NAME="MiniMaxAI/MiniMax-M2"
PORT=9084
HOST="0.0.0.0"
DTYPE="bfloat16"
MAX_MODEL_LEN=128000
TENSOR_PARALLEL_SIZE=4   # GPUs 0-3; GPUs 4-7 left free for other services
GPU_MEMORY_UTIL=0.95
MAX_NUM_SEQS=16

# Default generation parameters (caller can override via API)
TEMPERATURE=1.0
TOP_P=0.95
TOP_K=20
MAX_TOKENS=16384

# MiniMax M2 specific flags
ENABLE_AUTO_TOOL_CHOICE="--enable-auto-tool-choice"
TOOL_CALL_PARSER="--tool-call-parser minimax_m2"
REASONING_PARSER="--reasoning-parser minimax_m2"

# -----------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. CUDA may not be installed correctly.${NC}"
    exit 1
fi

# Check GPU count
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo -e "${GREEN}Detected $GPU_COUNT GPU(s)${NC}"

if [ "$GPU_COUNT" -lt "$TENSOR_PARALLEL_SIZE" ]; then
    echo -e "${RED}Error: Requested $TENSOR_PARALLEL_SIZE GPUs but only $GPU_COUNT available${NC}"
    exit 1
fi

# Show GPU info for the GPUs that will be used
echo -e "${GREEN}GPU Information (using first $TENSOR_PARALLEL_SIZE GPUs):${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -n $TENSOR_PARALLEL_SIZE

# Check port
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}Error: Port $PORT is already in use${NC}"
    echo "Please stop the existing service or choose a different port"
    exit 1
fi

# Check / prepare model directory
if [ -d "$MODEL_PATH" ] && [ "$(ls -A $MODEL_PATH)" ]; then
    echo -e "${GREEN}Using local model from: $MODEL_PATH${NC}"
    MODEL_ARG="$MODEL_PATH"
else
    echo -e "${YELLOW}Model not found locally. Will download from HuggingFace: $MODEL_NAME${NC}"
    echo -e "${YELLOW}This may take some time (~220GB download)...${NC}"
    echo -e "${YELLOW}Download will be cached to: $HF_HOME${NC}"
    MODEL_ARG="$MODEL_NAME"
    sudo mkdir -p "$MODEL_PATH"
    sudo chown -R "$(id -u):$(id -g)" "$MODEL_PATH"
fi

# Create logs directory relative to the script location
mkdir -p "${SCRIPT_DIR}/logs"

echo ""
echo -e "${GREEN}Starting vLLM server with the following configuration:${NC}"
echo "  Model:                   $MODEL_ARG"
echo "  Port:                    $PORT"
echo "  Host:                    $HOST"
echo "  Tensor Parallel Size:    $TENSOR_PARALLEL_SIZE GPUs (GPUs 0-3)"
echo "  Max Model Length:        $MAX_MODEL_LEN tokens (128K context)"
echo "  GPU Memory Utilization:  ${GPU_MEMORY_UTIL}"
echo "  Max Concurrent Seqs:     $MAX_NUM_SEQS"
echo "  Data Type:               $DTYPE"
echo ""
echo "  Default Generation Parameters (caller can override):"
echo "    Temperature: $TEMPERATURE"
echo "    Top-P:       $TOP_P"
echo "    Top-K:       $TOP_K"
echo "    Max Tokens:  $MAX_TOKENS"
echo ""
echo "  MiniMax M2 Features:"
echo "    - Auto Tool Choice: Enabled"
echo "    - Tool Call Parser: minimax_m2"
echo "    - Reasoning Parser: minimax_m2 (strips <think> tags)"
echo ""
echo -e "${YELLOW}Service will be accessible at:  http://localhost:$PORT${NC}"
echo -e "${YELLOW}Logs will be saved to:          ${SCRIPT_DIR}/logs/service.log${NC}"
echo -e "${YELLOW}Model uses ~220GB across $TENSOR_PARALLEL_SIZE GPUs${NC}"
echo ""
echo -e "${GREEN}Starting server... (initial model load may take a few minutes)${NC}"
echo ""

# Start vLLM server — log to both console and file
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
  2>&1 | tee "${SCRIPT_DIR}/logs/service.log"
