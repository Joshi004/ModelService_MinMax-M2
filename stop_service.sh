#!/bin/bash

# MiniMax M2.5 Service Stop Script
# This script gracefully stops the vLLM server for MiniMax-M2.5 model

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Stopping MiniMax M2.5 Service...${NC}"

# Function to check if service is running
check_service() {
    # Check for vLLM process serving MiniMax M2.5 model
    VLLM_PIDS=$(pgrep -f "vllm serve.*minimax-m2.5")
    if [ -z "$VLLM_PIDS" ]; then
        # Also check for the model name variant
        VLLM_PIDS=$(pgrep -f "vllm serve.*MiniMax-M2.5")
    fi
    echo "$VLLM_PIDS"
}

# Get running service PIDs
PIDS=$(check_service)

if [ -z "$PIDS" ]; then
    echo -e "${YELLOW}No MiniMax M2.5 service found running${NC}"
    exit 0
fi

echo -e "${YELLOW}Found MiniMax M2.5 service process(es): $PIDS${NC}"

# Gracefully terminate processes
echo -e "${GREEN}Sending SIGTERM for graceful shutdown...${NC}"
for PID in $PIDS; do
    echo "  Stopping process $PID..."
    kill -TERM $PID 2>/dev/null || true
done

# Wait for processes to terminate (max 30 seconds)
echo -e "${YELLOW}Waiting for processes to terminate...${NC}"
WAIT_TIME=0
MAX_WAIT=30

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    sleep 2
    WAIT_TIME=$((WAIT_TIME + 2))
    
    REMAINING_PIDS=$(check_service)
    if [ -z "$REMAINING_PIDS" ]; then
        echo -e "${GREEN}Service stopped successfully!${NC}"
        exit 0
    fi
    
    echo "  Still waiting... ($WAIT_TIME/${MAX_WAIT}s)"
done

# If processes still running, force kill
echo -e "${YELLOW}Processes did not terminate gracefully. Force killing...${NC}"
REMAINING_PIDS=$(check_service)

if [ ! -z "$REMAINING_PIDS" ]; then
    for PID in $REMAINING_PIDS; do
        echo "  Force killing process $PID..."
        kill -9 $PID 2>/dev/null || true
    done
    
    sleep 2
    
    # Final check
    FINAL_CHECK=$(check_service)
    if [ -z "$FINAL_CHECK" ]; then
        echo -e "${GREEN}Service stopped (force killed)${NC}"
        exit 0
    else
        echo -e "${RED}Error: Could not stop all processes${NC}"
        echo "Remaining PIDs: $FINAL_CHECK"
        exit 1
    fi
else
    echo -e "${GREEN}Service stopped successfully!${NC}"
    exit 0
fi

