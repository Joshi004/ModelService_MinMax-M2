# MiniMax M2 Service

A production-ready service for running the MiniMax-M2 model using vLLM's built-in API server.

## Overview

This service deploys the MiniMax-M2 model, a state-of-the-art Mixture of Experts (MoE) model specialized for coding, agentic workflows, and tool calling with advanced chain-of-thought reasoning capabilities.

### Model Information
- **Model**: MiniMax-M2
- **Architecture**: Mixture of Experts (MoE) - 230B total parameters, 10B active during inference
- **Location**: `/home/naresh/models/minimax-m2/`
- **Capabilities**: Advanced coding, tool calling, agentic workflows, chain-of-thought reasoning
- **Context Length**: 128K tokens
- **Specializations**: Multi-file code editing, coding-run-fix loops, test-validated repairs

### Hardware Configuration
- **GPUs**: 4x NVIDIA H100 (80GB each)
- **Tensor Parallelism**: Enabled across all 4 GPUs
- **Total GPU Memory**: 320GB
- **Model Requirements**: ~220GB for weights

## Quick Start

### 1. Start the Service

```bash
cd /home/naresh/minimax-m2-service
./start_service.sh
```

The startup script will:
- Activate the dedicated virtual environment
- Check GPU availability
- Load the model across 4 GPUs (first run may download ~220GB)
- Start the vLLM server on port 8084

**First Run Note**: If the model is not present locally, it will be automatically downloaded from HuggingFace. This is a ~220GB download and may take some time.

### 2. Test the Service

#### Quick Test

```bash
# Test basic completion
python test_client.py --test-type basic

# Test reasoning capabilities
python test_client.py --test-type reasoning

# Test tool calling
python test_client.py --test-type tools

# Run all tests
python test_client.py --test-type all
```

#### Manual Test with curl

```bash
curl http://localhost:8084/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "Write a Python function to reverse a linked list."
    }]
  }'
```

### 3. Stop the Service

```bash
./stop_service.sh
```

## API Usage

### Endpoint

```
POST http://localhost:8084/v1/chat/completions
```

### Request Format

The service provides an OpenAI-compatible API:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Your prompt here"
    }
  ],
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "max_tokens": 16384
}
```

### Generation Parameters

**Server Defaults** (used when not specified by caller):
- `temperature`: 1.0 (MiniMax M2 optimized)
- `top_p`: 0.95
- `top_k`: 20
- `max_tokens`: 16384

**Caller Override**: All parameters can be overridden in API requests. Caller-provided values always take precedence over server defaults.

### Response Format

```json
{
  "id": "cmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "MiniMaxAI/MiniMax-M2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Response with possible <think>reasoning</think> tags..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 500,
    "total_tokens": 650
  }
}
```

## Python Client Example

### Basic Completion

```python
import requests

def chat_with_minimax(prompt, temperature=None):
    """Chat with MiniMax M2 service."""
    
    payload = {
        "messages": [{"role": "user", "content": prompt}]
    }
    
    # Only include temperature if you want to override server default
    if temperature is not None:
        payload["temperature"] = temperature
    
    response = requests.post(
        "http://localhost:8084/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]

# Use server defaults (temp=1.0, top_p=0.95, top_k=20)
response = chat_with_minimax("Write a binary search function in Python.")
print(response)

# Override temperature for more deterministic output
response = chat_with_minimax(
    "Explain what this code does: def foo(x): return x**2",
    temperature=0.2
)
print(response)
```

### Tool Calling

```python
import requests
import json

def chat_with_tools(prompt, tools):
    """Chat with tool calling enabled."""
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools
    }
    
    response = requests.post(
        "http://localhost:8084/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    return response.json()

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    }
]

# Use tools
response = chat_with_tools(
    "Calculate the factorial of 10 using Python.",
    tools
)

# Check if model called a tool
if "tool_calls" in response["choices"][0]["message"]:
    for tool_call in response["choices"][0]["message"]["tool_calls"]:
        print(f"Tool: {tool_call['function']['name']}")
        print(f"Arguments: {tool_call['function']['arguments']}")
```

### Multi-turn Conversation

```python
import requests

def multi_turn_chat():
    """Example of multi-turn conversation."""
    
    messages = []
    
    # Turn 1
    messages.append({
        "role": "user",
        "content": "I need to build a REST API for a todo app. What endpoints should I create?"
    })
    
    response = requests.post(
        "http://localhost:8084/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={"messages": messages}
    ).json()
    
    # Add assistant response
    messages.append({
        "role": "assistant",
        "content": response["choices"][0]["message"]["content"]
    })
    
    # Turn 2
    messages.append({
        "role": "user",
        "content": "Can you show me the FastAPI code for the GET /todos endpoint?"
    })
    
    response = requests.post(
        "http://localhost:8084/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={"messages": messages}
    ).json()
    
    print(response["choices"][0]["message"]["content"])

multi_turn_chat()
```

## Configuration

### Environment Variables

Located in `config.env`:

```bash
# Model settings
MODEL_PATH="/home/naresh/models/minimax-m2"
MODEL_NAME="MiniMaxAI/MiniMax-M2"
PORT=8084
DTYPE="bfloat16"

# Performance
MAX_MODEL_LEN=128000        # Full 128K context
TENSOR_PARALLEL_SIZE=4      # 4x H100 GPUs
GPU_MEMORY_UTIL=0.95        # 95% GPU memory
MAX_NUM_SEQS=16            # Concurrent requests

# Server default generation parameters
# (Callers can override these in requests)
TEMPERATURE=1.0             # MiniMax M2 recommended
TOP_P=0.95
TOP_K=20
MAX_TOKENS=16384

# MiniMax M2 specific
SAFETENSORS_FAST_GPU=1                        # Fast model loading
ENABLE_AUTO_TOOL_CHOICE=true                  # Auto tool selection
TOOL_CALL_PARSER="minimax_m2"                 # Tool call parser
REASONING_PARSER="minimax_m2_append_think"    # Preserve <think> tags
```

### Understanding the Special Flags

#### `--enable-auto-tool-choice`
Enables automatic tool selection. The model intelligently decides when to use available tools based on the conversation context, without requiring explicit tool invocation instructions.

#### `--tool-call-parser minimax_m2`
Parses MiniMax M2's specific tool calling format. The model outputs tool calls in a structured JSON format that this parser correctly interprets and routes to the API response.

#### `--reasoning-parser minimax_m2_append_think`
Preserves the model's chain-of-thought reasoning in `<think>...</think>` tags. This is **critical** for maintaining performance:
- The model uses internal reasoning before generating answers
- Preserving this content in conversation history improves subsequent responses
- Removing thinking content will degrade multi-turn performance

Example with thinking tags:
```
<think>
To solve this problem, I need to:
1. First understand the data structure
2. Consider edge cases
3. Optimize for time complexity
</think>

Here's the solution...
```

#### `SAFETENSORS_FAST_GPU=1`
Enables optimized GPU memory loading for SafeTensors format model weights. This speeds up model initialization by 30-50%, reducing startup time from minutes to seconds with the 220GB model.

## Model Capabilities

### Coding Excellence

MiniMax M2 excels at:
- **Multi-file Edits**: Understanding and modifying code across multiple files
- **Code-Run-Fix Loops**: Iteratively debugging and fixing code
- **Test-Validated Repairs**: Generating fixes that pass test suites
- **Complex Refactoring**: Large-scale code restructuring

Example:
```python
prompt = """
I have a Python project with these files:
- main.py: Contains the Flask app
- database.py: Contains DB connection logic
- models.py: Contains SQLAlchemy models

The app is slow. Can you identify performance issues and suggest fixes?
"""
```

### Agentic Workflows

The model can plan and execute complex multi-step tasks:
- Break down complex problems into subtasks
- Decide when to use tools
- Self-correct based on intermediate results
- Maintain context across long interactions

### Tool Calling

Advanced tool calling with automatic selection:
- Model decides when tools are needed
- Generates properly formatted tool calls
- Handles tool results and continues reasoning
- Supports multiple tool calls in sequence

### Reasoning with `<think>` Tags

The model uses explicit chain-of-thought reasoning:
- Shows its thinking process in `<think>` tags
- Improves answer quality through deliberate reasoning
- Helps debugging by making logic transparent
- Critical for maintaining performance in conversations

**Important**: Always preserve `<think>` content in conversation history!

## Comparison: MiniMax M2 vs Qwen3-Omni

### MiniMax M2
- **Architecture**: MoE (230B total, 10B active) - Sparse activation
- **Specialization**: Coding, agentic tasks, tool calling
- **Modality**: Text-only
- **Context**: 128K tokens
- **Memory**: ~220GB model weights
- **vLLM**: Standard nightly builds
- **Throughput**: Higher (due to sparse MoE)
- **Best For**: Code generation, debugging, agentic workflows

### Qwen3-Omni
- **Architecture**: Dense (30B, all active)
- **Specialization**: Multimodal understanding
- **Modality**: Text, audio, video, image
- **Context**: 32K-64K tokens
- **Memory**: ~60GB model weights
- **vLLM**: Custom branch (qwen3-omni)
- **Throughput**: Lower (all parameters active)
- **Best For**: Video analysis, audio processing, multimodal tasks

### Key Differences

| Feature | MiniMax M2 | Qwen3-Omni |
|---------|-----------|------------|
| **Model Size** | 230B (10B active) | 30B (all active) |
| **Modalities** | Text only | Text, Audio, Video, Image |
| **Context Length** | 128K tokens | 32K-64K tokens |
| **Tool Calling** | Native support | Basic support |
| **Reasoning** | `<think>` tags | Chain-of-thought |
| **GPU Memory** | ~220GB | ~60GB |
| **GPUs Required** | 4x H100 | 2x H100 |
| **Port** | 8084 | 8002 |
| **vLLM Version** | Standard nightly | Custom branch |
| **Special Flags** | `--enable-auto-tool-choice`, `--tool-call-parser`, `--reasoning-parser` | `VLLM_USE_V1=0`, `--allowed-local-media-path` |

### When to Use Which Model

**Use MiniMax M2 when:**
- Building or debugging code
- Need tool calling and agentic workflows
- Working with long contexts (up to 128K)
- Need explicit reasoning steps
- Building AI agents

**Use Qwen3-Omni when:**
- Processing videos or audio
- Need multimodal understanding
- Analyzing visual content
- Real-time streaming responses
- Video/audio captioning

## Service Management

### Check Service Status

```bash
# Check if service is running
curl http://localhost:8084/health

# Check vLLM processes
ps aux | grep "vllm serve.*minimax"

# Check port usage
lsof -i :8084
```

### View Logs

```bash
# View logs in real-time
tail -f /home/naresh/minimax-m2-service/logs/service.log

# View last 100 lines
tail -n 100 /home/naresh/minimax-m2-service/logs/service.log

# Search for errors
grep -i error /home/naresh/minimax-m2-service/logs/service.log
```

### GPU Monitoring

```bash
# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Check all 4 GPUs
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

# Use nvtop for interactive monitoring (if installed)
nvtop
```

## Troubleshooting

### Service Won't Start

**1. Check GPU availability**
```bash
nvidia-smi
```
Ensure all 4 H100 GPUs are visible and healthy.

**2. Verify virtual environment**
```bash
source /home/naresh/venvs/minimax-m2-service/bin/activate
which vllm
vllm --version
```

**3. Check model path**
```bash
ls -la /home/naresh/models/minimax-m2/
```
If empty, the model will be downloaded automatically on first run.

**4. Port already in use**
```bash
lsof -i :8084
```
Stop the conflicting process or change the port in `config.env`.

### Out of Memory Errors

**1. Reduce max model length**
Edit `start_service.sh`:
```bash
MAX_MODEL_LEN=65536  # Instead of 128000
```

**2. Reduce GPU memory utilization**
```bash
GPU_MEMORY_UTIL=0.90  # Instead of 0.95
```

**3. Reduce concurrent sequences**
```bash
MAX_NUM_SEQS=8  # Instead of 16
```

**4. Verify tensor parallelism**
Make sure `TENSOR_PARALLEL_SIZE=4` matches available GPUs.

### Model Loading Slowly

**1. Check SAFETENSORS_FAST_GPU**
Ensure it's set in `start_service.sh`:
```bash
export SAFETENSORS_FAST_GPU=1
```

**2. Check storage speed**
Model loading speed depends on disk I/O. Consider using NVMe storage for model weights.

**3. First-time download**
Initial model download (~220GB) will take time. Subsequent startups should be faster.

### Responses Don't Include `<think>` Tags

**1. Verify reasoning parser**
Check `start_service.sh` includes:
```bash
--reasoning-parser minimax_m2_append_think
```

**2. Check vLLM version**
Ensure you're using the latest nightly build:
```bash
source /home/naresh/venvs/minimax-m2-service/bin/activate
pip install --upgrade --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
```

### Tool Calling Not Working

**1. Verify tool call parser**
Check `start_service.sh` includes:
```bash
--tool-call-parser minimax_m2
--enable-auto-tool-choice
```

**2. Check tool definitions**
Ensure tools are properly formatted in the API request with `type: "function"` and proper schema.

### Connection Refused

**1. Ensure service is running**
```bash
ps aux | grep "vllm serve"
```

**2. Check port**
```bash
netstat -tuln | grep 8084
```

**3. Verify firewall**
If accessing remotely, ensure port 8084 is open.

## Performance Tips

1. **Optimal Context Length**: Best performance with contexts under 64K tokens
2. **Batch Processing**: Process multiple requests concurrently (up to MAX_NUM_SEQS)
3. **GPU Utilization**: Monitor with `nvidia-smi` to ensure all 4 GPUs are utilized evenly
4. **Temperature Tuning**: 
   - Use 1.0 (default) for creative/exploratory tasks
   - Use 0.1-0.3 for deterministic code generation
   - Use 0.5-0.8 for balanced responses
5. **Preserve Thinking**: Always include `<think>` content in conversation history
6. **Tool Calling**: Let the model decide when to use tools (auto-tool-choice enabled)

## Directory Structure

```
minimax-m2-service/
├── start_service.sh      # Start the vLLM server
├── stop_service.sh       # Stop the service
├── config.env           # Configuration variables
├── config.env.example   # Example configuration
├── test_client.py       # Test client script
├── README.md           # This file
└── logs/               # Service logs
    └── service.log     # Main service log
```

## Dependencies

Installed in virtual environment at `/home/naresh/venvs/minimax-m2-service/`:

- vLLM 0.11.2 (nightly)
- PyTorch 2.9.0 with CUDA 12.9
- Transformers 4.57.1
- xformers 0.0.33.post1
- flashinfer-python 0.5.2
- Other dependencies (see venv)

## Model Information & License

- **Model**: [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)
- **Paper**: Check HuggingFace model card for publication details
- **License**: See model repository for license information
- **Support**: For model-specific issues, refer to the HuggingFace repository

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [MiniMax M2 Model Card](https://huggingface.co/MiniMaxAI/MiniMax-M2)
- [MiniMax M2 vLLM Deployment Guide](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/docs/vllm_deploy_guide.md)
- [Tool Calling Guide](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/docs/tool_calling_guide.md)

---

**Note**: This is a production deployment. Always monitor GPU usage and service logs for optimal performance. The model requires significant compute resources (4x H100 GPUs) and is optimized for coding and agentic workflows.

