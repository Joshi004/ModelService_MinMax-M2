# MiniMax M2 Service - Setup Complete ✅

The MiniMax M2 service has been successfully set up and is ready to use!

## Setup Summary

### ✅ Completed Tasks

1. **Directory Structure**: Created service directory at `/home/naresh/minimax-m2-service/`
2. **Virtual Environment**: Set up dedicated venv at `/home/naresh/venvs/minimax-m2-service/`
3. **Dependencies**: Installed vLLM 0.11.2 and all required packages
4. **Configuration Files**: Created `config.env` and `config.env.example`
5. **Service Scripts**: Created `start_service.sh` and `stop_service.sh`
6. **Test Client**: Created comprehensive `test_client.py`
7. **Documentation**: Created detailed `README.md`

### 📋 Service Configuration

- **Model**: MiniMaxAI/MiniMax-M2
- **Model Path**: `/home/naresh/models/minimax-m2/`
- **Port**: 8084
- **GPUs**: 4x H100 (80GB each)
- **Context Length**: 128K tokens
- **Virtual Environment**: `/home/naresh/venvs/minimax-m2-service/`

### 🔧 Configuration Highlights

**Server Default Parameters** (caller can override):
- Temperature: 1.0 (MiniMax M2 optimized)
- Top-P: 0.95
- Top-K: 20
- Max Tokens: 16384

**MiniMax M2 Specific Features**:
- ✅ Auto tool choice enabled
- ✅ Tool call parser: minimax_m2
- ✅ Reasoning parser: minimax_m2_append_think (preserves `<think>` tags)
- ✅ Fast GPU loading enabled (SAFETENSORS_FAST_GPU=1)

## Next Steps

### 1. Start the Service

```bash
cd /home/naresh/minimax-m2-service
./start_service.sh
```

**Important Notes**:
- **First Run**: The model (~220GB) will be automatically downloaded from HuggingFace if not present
- **Startup Time**: Initial model loading may take 5-10 minutes
- **GPU Check**: The script will verify all 4 GPUs are available before starting
- **Logs**: Output will be saved to `logs/service.log`

### 2. Verify the Service

Once started, verify the service is running:

```bash
# Check service health
curl http://localhost:8084/health

# Or use the test client
python test_client.py --test-type basic
```

### 3. Run Comprehensive Tests

```bash
# Test basic completion
python test_client.py --test-type basic

# Test reasoning with <think> tags
python test_client.py --test-type reasoning

# Test tool calling
python test_client.py --test-type tools

# Test parameter override
python test_client.py --test-type params

# Run all tests
python test_client.py --test-type all
```

### 4. Monitor GPU Usage

While the service is running, monitor GPU utilization:

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Detailed GPU info
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

## Usage Examples

### Basic Completion (Using Server Defaults)

```bash
curl http://localhost:8084/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "Write a Python function to implement quicksort."
    }]
  }'
```

The server will use default parameters: temp=1.0, top_p=0.95, top_k=20

### Override Parameters

```bash
curl http://localhost:8084/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "Explain what this function does: def f(n): return n if n<2 else f(n-1)+f(n-2)"
    }],
    "temperature": 0.2,
    "max_tokens": 500
  }'
```

### Python Client

```python
import requests

def chat(prompt, **kwargs):
    """Chat with MiniMax M2. Kwargs override server defaults."""
    payload = {"messages": [{"role": "user", "content": prompt}]}
    payload.update(kwargs)
    
    response = requests.post(
        "http://localhost:8084/v1/chat/completions",
        json=payload
    )
    return response.json()["choices"][0]["message"]["content"]

# Use server defaults
response = chat("Write a binary search function.")

# Override temperature for more deterministic output
response = chat("Explain bubble sort.", temperature=0.1)
```

## Service Management

### Start Service
```bash
cd /home/naresh/minimax-m2-service
./start_service.sh
```

### Stop Service
```bash
cd /home/naresh/minimax-m2-service
./stop_service.sh
```

### View Logs
```bash
tail -f /home/naresh/minimax-m2-service/logs/service.log
```

### Check Status
```bash
# Check if running
ps aux | grep "vllm serve.*minimax"

# Check port
lsof -i :8084

# Test health
curl http://localhost:8084/health
```

## Model Comparison

### MiniMax M2 (This Service - Port 8084)
- **Specialization**: Coding, agentic workflows, tool calling
- **Modality**: Text-only
- **Context**: 128K tokens
- **GPUs**: 4x H100
- **Memory**: ~220GB
- **Best For**: Code generation, debugging, multi-file edits, tool use

### Qwen3-Omni (Port 8002)
- **Specialization**: Multimodal understanding
- **Modality**: Text, audio, video, image
- **Context**: 32K-64K tokens
- **GPUs**: 2x H100
- **Memory**: ~60GB
- **Best For**: Video/audio analysis, multimodal tasks

### Qwen3-Captioner (Port 8003)
- **Specialization**: Audio captioning
- **Modality**: Audio input, text output
- **Context**: 32K tokens
- **GPUs**: 2x H100
- **Memory**: ~60GB
- **Best For**: Audio transcription, detailed audio descriptions

## Troubleshooting

### Service Won't Start

1. **Check GPU availability**:
   ```bash
   nvidia-smi
   ```
   Ensure 4 GPUs are visible and healthy.

2. **Check port availability**:
   ```bash
   lsof -i :8084
   ```
   If port is in use, stop the conflicting service or change port in `config.env`.

3. **Check virtual environment**:
   ```bash
   source /home/naresh/venvs/minimax-m2-service/bin/activate
   which vllm
   ```

4. **Check logs**:
   ```bash
   tail -100 /home/naresh/minimax-m2-service/logs/service.log
   ```

### Model Download

On first run, the model will be downloaded:
- **Size**: ~220GB
- **Time**: Depends on network speed (30min - 2hrs typically)
- **Location**: `/home/naresh/models/minimax-m2/`
- **Progress**: Shown in terminal output

To pre-download the model:
```bash
source /home/naresh/venvs/minimax-m2-service/bin/activate
huggingface-cli download MiniMaxAI/MiniMax-M2 --local-dir /home/naresh/models/minimax-m2
```

### Memory Issues

If you encounter OOM errors:

1. **Reduce max context length** (edit `start_service.sh`):
   ```bash
   MAX_MODEL_LEN=65536  # Instead of 128000
   ```

2. **Reduce concurrent requests**:
   ```bash
   MAX_NUM_SEQS=8  # Instead of 16
   ```

3. **Reduce GPU memory utilization**:
   ```bash
   GPU_MEMORY_UTIL=0.90  # Instead of 0.95
   ```

## Key Features

### 1. Parameter Override System
- **Server provides sensible defaults**: temp=1.0, top_p=0.95, top_k=20
- **Callers can override**: Specify parameters in API request
- **Flexible**: Use defaults for most cases, override for specific needs

### 2. Chain-of-Thought Reasoning
- Model uses `<think>...</think>` tags for internal reasoning
- Reasoning parser preserves this content in responses
- **Important**: Keep thinking content in conversation history for best performance

### 3. Tool Calling
- Automatic tool selection enabled
- Model decides when to use tools based on context
- Supports complex multi-tool workflows

### 4. Long Context Support
- Full 128K token context window
- Handles large codebases and documents
- Efficient MoE architecture (only 10B active params)

## Files Created

```
/home/naresh/minimax-m2-service/
├── README.md                 # Comprehensive documentation
├── SETUP_COMPLETE.md        # This file
├── config.env               # Configuration (with your paths)
├── config.env.example       # Example configuration
├── start_service.sh         # Service startup script (executable)
├── stop_service.sh          # Service shutdown script (executable)
├── test_client.py           # Test client (executable)
└── logs/                    # Log directory
    └── service.log          # Will be created on first run

/home/naresh/venvs/minimax-m2-service/
└── (virtual environment with vLLM 0.11.2 and dependencies)

/home/naresh/models/minimax-m2/
└── (model will be downloaded here on first run)
```

## Resources

- **Service README**: `/home/naresh/minimax-m2-service/README.md`
- **Configuration**: `/home/naresh/minimax-m2-service/config.env`
- **Model Card**: https://huggingface.co/MiniMaxAI/MiniMax-M2
- **vLLM Docs**: https://docs.vllm.ai/
- **Tool Calling Guide**: https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/docs/tool_calling_guide.md

## Summary

✅ **All setup tasks completed successfully!**

The MiniMax M2 service is ready to:
- Generate high-quality code
- Perform complex debugging and refactoring
- Use tools automatically when needed
- Reason through problems with `<think>` tags
- Handle long contexts up to 128K tokens
- Process multiple concurrent requests efficiently

You can now start the service and begin using MiniMax M2 for your coding and agentic workflow needs!

---

**Need Help?**
- Check `README.md` for detailed usage instructions
- View logs at `logs/service.log` for debugging
- Run `python test_client.py --test-type all` for comprehensive testing

