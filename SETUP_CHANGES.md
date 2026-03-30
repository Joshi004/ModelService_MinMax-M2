# MiniMax-M2.5 Service — Changes Applied

## System Facts (This Machine)

| Item | Value |
|------|-------|
| Current user | `vision` — scripts now use `${HOME}` (dynamic) |
| Home directory | `/home/vision` |
| GPUs | 8× NVIDIA H100 80GB HBM3 — **640 GB total VRAM** |
| CUDA toolkit | `12.4` at `/usr/local/cuda-12.4` (symlinked: `/usr/local/cuda`) |
| CUDA driver | 570.211.01 |
| Root disk `/` | ~535 GB free |
| Data disk `/mnt/hf-cache` | ~467 GB free — model weights stored here |

---

## Resource Assessment

| Requirement | Needed | Available | Status |
|-------------|--------|-----------|--------|
| GPU VRAM (4-GPU config) | ~220 GB | 320 GB (4× H100) | ✅ |
| Model disk space | ~220 GB | 467 GB on `/mnt/hf-cache` | ✅ |
| Python venv disk space | ~10 GB | 535 GB on `/` | ✅ |

**GPU assignment**: MiniMax-M2.5 uses GPUs 0-3. GPUs 4-7 are free for Parakeet and future services.

---

## Changes Applied

### `config.env`

| Variable | Old value | New value |
|----------|-----------|-----------|
| `CUDA_HOME` | `/usr/local/cuda-12.9` | `/usr/local/cuda` |
| `MODEL_PATH` | `/home/naresh/models/minimax-m2` | `/mnt/hf-cache/models/minimax-m2.5` |
| `MODEL_NAME` | `MiniMaxAI/MiniMax-M2` | `MiniMaxAI/MiniMax-M2.5` |
| `PORT` | `7104` | `9084` |
| `TENSOR_PARALLEL_SIZE` | `4` | `4` (unchanged — GPUs 0-3) |
| `HF_HOME` | _(missing)_ | `/mnt/hf-cache/huggingface` |
| `HUGGINGFACE_HUB_CACHE` | _(missing)_ | `/mnt/hf-cache/huggingface/hub` |

### `config.env.example`

Same changes as `config.env` to keep the template in sync.

### `start_service.sh`

| Item | Old | New |
|------|-----|-----|
| `CUDA_HOME` | `/usr/local/cuda-12.9` | `/usr/local/cuda` |
| Venv source | `/home/naresh/venvs/minimax-m2-service/bin/activate` | `${HOME}/venvs/minimax-m2.5-service/bin/activate` |
| Auto-venv creation | _(missing)_ | Added — creates venv and installs vLLM if not present |
| `MODEL_PATH` | `/home/naresh/models/minimax-m2` | `/mnt/hf-cache/models/minimax-m2.5` |
| `MODEL_NAME` | `MiniMaxAI/MiniMax-M2` | `MiniMaxAI/MiniMax-M2.5` |
| `PORT` | hardcoded `7104` | `9084` |
| `TOP_K` | `20` | `40` |
| `MAX_NUM_SEQS` | `16` | `10` |
| Logs mkdir | `/home/naresh/minimax-m2-service/logs` | `${SCRIPT_DIR}/logs` |
| Tee log path | `/home/naresh/minimax-m2-service/logs/service.log` | `${SCRIPT_DIR}/logs/service.log` |
| `HF_HOME` export | _(missing)_ | `/mnt/hf-cache/huggingface` |
| `HUGGINGFACE_HUB_CACHE` export | _(missing)_ | `/mnt/hf-cache/huggingface/hub` |
| `SCRIPT_DIR` | _(missing)_ | `$(cd "$(dirname "$0")" && pwd)` |

### `verify_setup.sh`

All hardcoded `/home/naresh/` paths replaced with `${SCRIPT_DIR}` or `${HOME}/...`.
Port check updated from `8084` to `9084`.
Model directory check updated to `/mnt/hf-cache/models/minimax-m2.5`.
Venv path updated to `~/venvs/minimax-m2.5-service`.
HuggingFace cache directory check added.

### `test_client.py`

| Item | Old | New |
|------|-----|-----|
| Default port (argparse) | `7104` | `9084` |
| Default URL in `__init__` | `http://localhost:8084` | `http://localhost:9084` |
| Error message path | `/home/naresh/minimax-m2-service` | `~/ModelService_MinMax-M2` |

### `test_curl.sh`

| Item | Old | New |
|------|-----|-----|
| Default `BASE_URL` | `http://localhost:8004` | `http://localhost:9084` |

---

## Files Not Changed

| File | Reason |
|------|--------|
| `stop_service.sh` | No hardcoded paths — uses `pgrep` pattern matching only |

---

## How to Use

### First Time — Download Model Then Start

```bash
cd ~/ModelService_MinMax-M2

# Step 1: Run start script once to create venv + install vLLM
# (Ctrl+C once it tries to start the server)
./start_service.sh

# Step 2: Download the model (~220 GB)
source ${HOME}/venvs/minimax-m2.5-service/bin/activate
huggingface-cli download MiniMaxAI/MiniMax-M2.5 \
    --local-dir /mnt/hf-cache/models/minimax-m2.5

# Step 3: Start service (model now present locally)
./start_service.sh
```

### Subsequent Starts

```bash
cd ~/ModelService_MinMax-M2
./start_service.sh
```

### Health Check

```bash
curl http://localhost:9084/health
```

### Run Tests

```bash
cd ~/ModelService_MinMax-M2
python test_client.py --test-type all
```
