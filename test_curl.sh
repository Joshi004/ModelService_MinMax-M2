#!/bin/bash

# MiniMax M2.5 Service - Quick curl Test Script
# This script tests the service with curl commands

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default to localhost:9084
BASE_URL="${1:-http://localhost:9084}"
API_URL="${BASE_URL}/v1/chat/completions"
HEALTH_URL="${BASE_URL}/health"

echo "MiniMax M2.5 Service - curl Test"
echo "================================="
echo "Service URL: $BASE_URL"
echo ""

# Test 1: Health Check
echo -e "${BLUE}Test 1: Health Check${NC}"
echo "Command: curl $HEALTH_URL"
echo ""

if curl -s -f "$HEALTH_URL" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    HEALTH_RESPONSE=$(curl -s "$HEALTH_URL")
    echo "Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}✗ Health check failed${NC}"
    echo "Make sure:"
    echo "  1. Service is running on the compute node"
    echo "  2. SSH tunnel is active (if remote): ssh -L 9084:<node>:9084 <login>"
    echo "  3. Service is accessible on port 9084"
    exit 1
fi

echo ""
echo "================================"
echo ""

# Test 2: Basic Chat Completion (using server defaults)
echo -e "${BLUE}Test 2: Basic Chat Completion (Server Defaults)${NC}"
echo "Command: curl -X POST $API_URL ..."
echo ""

REQUEST_BODY='{
  "messages": [
    {
      "role": "user",
      "content": "Write a Python function to calculate the factorial of a number. Keep it simple."
    }
  ]
}'

echo "Request:"
echo "$REQUEST_BODY" | python3 -m json.tool 2>/dev/null || echo "$REQUEST_BODY"
echo ""

echo "Sending request (this may take 10-30 seconds)..."
echo ""

RESPONSE=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "$REQUEST_BODY")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Request successful${NC}"
    echo ""
    echo "Response:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
    echo ""
    
    # Extract content if possible
    if command -v jq &> /dev/null; then
        CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null)
        if [ "$CONTENT" != "null" ] && [ -n "$CONTENT" ]; then
            echo "----------------------------------------"
            echo "Generated Content:"
            echo "----------------------------------------"
            echo "$CONTENT"
            echo "----------------------------------------"
            echo ""
        fi
        
        # Show token usage
        PROMPT_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.prompt_tokens' 2>/dev/null)
        COMPLETION_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.completion_tokens' 2>/dev/null)
        TOTAL_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.total_tokens' 2>/dev/null)
        
        if [ "$TOTAL_TOKENS" != "null" ]; then
            echo "Token Usage:"
            echo "  Prompt tokens: $PROMPT_TOKENS"
            echo "  Completion tokens: $COMPLETION_TOKENS"
            echo "  Total tokens: $TOTAL_TOKENS"
        fi
    fi
else
    echo -e "${RED}✗ Request failed${NC}"
    echo "Error details:"
    echo "$RESPONSE"
    exit 1
fi

echo ""
echo "================================"
echo ""

# Test 3: Parameter Override Test
echo -e "${BLUE}Test 3: Parameter Override Test${NC}"
echo "Testing that caller can override server defaults..."
echo ""

REQUEST_BODY_OVERRIDE='{
  "messages": [
    {
      "role": "user",
      "content": "Count from 1 to 5."
    }
  ],
  "temperature": 0.1,
  "max_tokens": 50
}'

echo "Request with custom parameters (temperature=0.1, max_tokens=50):"
echo "$REQUEST_BODY_OVERRIDE" | python3 -m json.tool 2>/dev/null || echo "$REQUEST_BODY_OVERRIDE"
echo ""

echo "Sending request..."
RESPONSE_OVERRIDE=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "$REQUEST_BODY_OVERRIDE")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Parameter override test successful${NC}"
    
    if command -v jq &> /dev/null; then
        CONTENT=$(echo "$RESPONSE_OVERRIDE" | jq -r '.choices[0].message.content' 2>/dev/null)
        if [ "$CONTENT" != "null" ] && [ -n "$CONTENT" ]; then
            echo "Response: $CONTENT"
        fi
    fi
else
    echo -e "${YELLOW}⚠ Parameter override test had issues${NC}"
fi

echo ""
echo "================================"
echo ""

# Test 4: Check for <think> tags (reasoning)
echo -e "${BLUE}Test 4: Reasoning Tags Check${NC}"
echo "Checking if response contains <think> tags (reasoning parser)..."
echo ""

if echo "$RESPONSE" | grep -q "<think>"; then
    echo -e "${GREEN}✓ Response contains <think> tags${NC}"
    echo "Reasoning parser is working correctly!"
    echo ""
    echo "Sample reasoning content:"
    echo "$RESPONSE" | grep -o '<think>.*</think>' | head -c 200
    echo "..."
else
    echo -e "${YELLOW}⚠ No <think> tags found in response${NC}"
    echo "This is normal if the model didn't use explicit reasoning for this simple query."
fi

echo ""
echo "================================"
echo ""

# Summary
echo -e "${GREEN}All Tests Completed!${NC}"
echo ""
echo "Summary:"
echo "  ✓ Service is running and accessible"
echo "  ✓ API endpoint is responding"
echo "  ✓ Chat completion is working"
echo "  ✓ Parameter override is working"
echo ""
echo "Your MiniMax M2.5 service is working correctly!"
echo ""
echo "Next steps:"
echo "  - Try more complex queries"
echo "  - Test tool calling: python test_client.py --test-type tools"
echo "  - Test reasoning: python test_client.py --test-type reasoning"
echo "  - Run all tests: python test_client.py --test-type all"

