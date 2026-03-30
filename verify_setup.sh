#!/bin/bash

# MiniMax M2.5 Service Setup Verification Script
# This script verifies that the service is correctly set up

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Resolve the directory this script lives in
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "MiniMax M2.5 Service Setup Verification"
echo "========================================"
echo ""

ERRORS=0
WARNINGS=0

# Check 1: Directory structure
echo "Checking directory structure..."
if [ -d "${SCRIPT_DIR}" ]; then
    echo -e "${GREEN}✓${NC} Service directory exists: ${SCRIPT_DIR}"
else
    echo -e "${RED}✗${NC} Service directory missing: ${SCRIPT_DIR}"
    ERRORS=$((ERRORS + 1))
fi

if [ -d "${SCRIPT_DIR}/logs" ]; then
    echo -e "${GREEN}✓${NC} Logs directory exists"
else
    echo -e "${YELLOW}⚠${NC} Logs directory not yet created (will be created on first start)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 2: Required files
echo ""
echo "Checking required files..."
FILES=(
    "start_service.sh"
    "stop_service.sh"
    "config.env"
    "config.env.example"
    "test_client.py"
    "README.md"
    "SETUP_COMPLETE.md"
)

for file in "${FILES[@]}"; do
    if [ -f "${SCRIPT_DIR}/$file" ]; then
        echo -e "${GREEN}✓${NC} $file exists"
    else
        echo -e "${RED}✗${NC} $file missing"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check 3: Executable permissions
echo ""
echo "Checking executable permissions..."
EXECUTABLES=(
    "start_service.sh"
    "stop_service.sh"
    "test_client.py"
)

for file in "${EXECUTABLES[@]}"; do
    if [ -x "${SCRIPT_DIR}/$file" ]; then
        echo -e "${GREEN}✓${NC} $file is executable"
    else
        echo -e "${YELLOW}⚠${NC} $file is not executable"
        WARNINGS=$((WARNINGS + 1))
    fi
done

# Check 4: Virtual environment
echo ""
echo "Checking virtual environment..."
VENV_PATH="${HOME}/venvs/minimax-m2.5-service"

if [ -d "${VENV_PATH}" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment exists: ${VENV_PATH}"

    if [ -f "${VENV_PATH}/bin/activate" ]; then
        echo -e "${GREEN}✓${NC} Virtual environment is valid"

        # Check vLLM installation
        source "${VENV_PATH}/bin/activate"
        if python -c "import vllm" 2>/dev/null; then
            VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null)
            echo -e "${GREEN}✓${NC} vLLM is installed (version $VLLM_VERSION)"
        else
            echo -e "${RED}✗${NC} vLLM is not installed in the venv"
            ERRORS=$((ERRORS + 1))
        fi
        deactivate 2>/dev/null || true
    else
        echo -e "${RED}✗${NC} Virtual environment is corrupted"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${YELLOW}⚠${NC} Virtual environment not yet created (will be auto-created on first start)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 5: Model directory on data disk
echo ""
echo "Checking model directory..."
MODEL_DIR="/mnt/hf-cache/models"
MODEL_PATH="/mnt/hf-cache/models/minimax-m2.5"

if [ -d "${MODEL_DIR}" ]; then
    echo -e "${GREEN}✓${NC} Models directory exists: ${MODEL_DIR}"

    if [ -d "${MODEL_PATH}" ]; then
        if [ "$(ls -A ${MODEL_PATH} 2>/dev/null)" ]; then
            echo -e "${GREEN}✓${NC} MiniMax M2.5 model is downloaded"
        else
            echo -e "${YELLOW}⚠${NC} MiniMax M2.5 model directory is empty (will download on first run)"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo -e "${YELLOW}⚠${NC} MiniMax M2.5 model not yet downloaded (will be created on first run)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${YELLOW}⚠${NC} Models directory not found at ${MODEL_DIR} (will be created on first run)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 6: HuggingFace cache on data disk
echo ""
echo "Checking HuggingFace cache directory..."
if [ -d "/mnt/hf-cache/huggingface" ]; then
    echo -e "${GREEN}✓${NC} HuggingFace cache directory exists: /mnt/hf-cache/huggingface"
else
    echo -e "${YELLOW}⚠${NC} HuggingFace cache directory not yet created (will be created on first run)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 7: CUDA/GPU
echo ""
echo "Checking CUDA/GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -ge 4 ]; then
        echo -e "${GREEN}✓${NC} Found $GPU_COUNT GPUs (4 required for MiniMax M2.5)"
        echo ""
        nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null | head -n 5
    elif [ "$GPU_COUNT" -gt 0 ]; then
        echo -e "${YELLOW}⚠${NC} Found $GPU_COUNT GPUs (4 required for optimal performance)"
        WARNINGS=$((WARNINGS + 1))
    else
        echo -e "${YELLOW}⚠${NC} No GPUs detected"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${YELLOW}⚠${NC} nvidia-smi not found"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 8: Port availability
echo ""
echo "Checking port availability..."
if command -v lsof &> /dev/null; then
    if lsof -Pi :9084 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${YELLOW}⚠${NC} Port 9084 is already in use"
        WARNINGS=$((WARNINGS + 1))
    else
        echo -e "${GREEN}✓${NC} Port 9084 is available"
    fi
else
    echo -e "${YELLOW}⚠${NC} lsof not available, cannot check port"
    WARNINGS=$((WARNINGS + 1))
fi

# Summary
echo ""
echo "========================================"
echo "Verification Summary"
echo "========================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Your MiniMax M2.5 service is ready to start."
    echo ""
    echo "Next steps:"
    echo "  1. cd ${SCRIPT_DIR}"
    echo "  2. ./start_service.sh"
    echo "  3. Wait for model loading (5-10 minutes on first run)"
    echo "  4. python test_client.py --test-type all"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Setup complete with $WARNINGS warning(s)${NC}"
    echo ""
    echo "The service should work, but you may want to address the warnings above."
    echo ""
    echo "You can start the service with:"
    echo "  cd ${SCRIPT_DIR}"
    echo "  ./start_service.sh"
    exit 0
else
    echo -e "${RED}✗ Setup incomplete with $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before starting the service."
    exit 1
fi

