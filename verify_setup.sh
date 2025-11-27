#!/bin/bash

# MiniMax M2 Service Setup Verification Script
# This script verifies that the service is correctly set up

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "MiniMax M2 Service Setup Verification"
echo "======================================"
echo ""

ERRORS=0
WARNINGS=0

# Check 1: Directory structure
echo "Checking directory structure..."
if [ -d "/home/naresh/minimax-m2-service" ]; then
    echo -e "${GREEN}✓${NC} Service directory exists"
else
    echo -e "${RED}✗${NC} Service directory missing"
    ERRORS=$((ERRORS + 1))
fi

if [ -d "/home/naresh/minimax-m2-service/logs" ]; then
    echo -e "${GREEN}✓${NC} Logs directory exists"
else
    echo -e "${RED}✗${NC} Logs directory missing"
    ERRORS=$((ERRORS + 1))
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
    if [ -f "/home/naresh/minimax-m2-service/$file" ]; then
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
    if [ -x "/home/naresh/minimax-m2-service/$file" ]; then
        echo -e "${GREEN}✓${NC} $file is executable"
    else
        echo -e "${YELLOW}⚠${NC} $file is not executable"
        WARNINGS=$((WARNINGS + 1))
    fi
done

# Check 4: Virtual environment
echo ""
echo "Checking virtual environment..."
if [ -d "/home/naresh/venvs/minimax-m2-service" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment exists"
    
    if [ -f "/home/naresh/venvs/minimax-m2-service/bin/activate" ]; then
        echo -e "${GREEN}✓${NC} Virtual environment is valid"
        
        # Check vLLM installation
        source /home/naresh/venvs/minimax-m2-service/bin/activate
        if python -c "import vllm" 2>/dev/null; then
            VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null)
            echo -e "${GREEN}✓${NC} vLLM is installed (version $VLLM_VERSION)"
        else
            echo -e "${RED}✗${NC} vLLM is not installed"
            ERRORS=$((ERRORS + 1))
        fi
        deactivate 2>/dev/null || true
    else
        echo -e "${RED}✗${NC} Virtual environment is corrupted"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${RED}✗${NC} Virtual environment missing"
    ERRORS=$((ERRORS + 1))
fi

# Check 5: Model directory
echo ""
echo "Checking model directory..."
if [ -d "/home/naresh/models" ]; then
    echo -e "${GREEN}✓${NC} Models directory exists"
    
    if [ -d "/home/naresh/models/minimax-m2" ]; then
        if [ "$(ls -A /home/naresh/models/minimax-m2 2>/dev/null)" ]; then
            echo -e "${GREEN}✓${NC} MiniMax M2 model is downloaded"
        else
            echo -e "${YELLOW}⚠${NC} MiniMax M2 model directory is empty (will download on first run)"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo -e "${YELLOW}⚠${NC} MiniMax M2 model directory not created (will be created on first run)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${YELLOW}⚠${NC} Models directory doesn't exist (will be created on first run)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 6: CUDA/GPU (if available)
echo ""
echo "Checking CUDA/GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -ge 4 ]; then
        echo -e "${GREEN}✓${NC} Found $GPU_COUNT GPUs (4 required)"
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
    echo -e "${YELLOW}⚠${NC} nvidia-smi not found (may not be on GPU machine)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 7: Port availability
echo ""
echo "Checking port availability..."
if command -v lsof &> /dev/null; then
    if lsof -Pi :8084 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${YELLOW}⚠${NC} Port 8084 is already in use"
        WARNINGS=$((WARNINGS + 1))
    else
        echo -e "${GREEN}✓${NC} Port 8084 is available"
    fi
else
    echo -e "${YELLOW}⚠${NC} lsof not available, cannot check port"
    WARNINGS=$((WARNINGS + 1))
fi

# Summary
echo ""
echo "======================================"
echo "Verification Summary"
echo "======================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Your MiniMax M2 service is ready to start."
    echo ""
    echo "Next steps:"
    echo "  1. cd /home/naresh/minimax-m2-service"
    echo "  2. ./start_service.sh"
    echo "  3. Wait for model loading (5-10 minutes)"
    echo "  4. python test_client.py --test-type all"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Setup complete with $WARNINGS warning(s)${NC}"
    echo ""
    echo "The service should work, but you may want to address the warnings above."
    echo ""
    echo "You can start the service with:"
    echo "  cd /home/naresh/minimax-m2-service"
    echo "  ./start_service.sh"
    exit 0
else
    echo -e "${RED}✗ Setup incomplete with $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before starting the service."
    exit 1
fi

