# MiniMax M2 Service - Quick Start Guide

## 🚀 Fastest Way to Start

### Option 1: Interactive Session (Recommended for Testing)

```bash
# Request 4 GPUs
srun --gres=gpu:4 --cpus-per-task=16 --mem=500G --time=24:00:00 --pty bash

# Once allocated, start service
cd /home/naresh/minimax-m2-service
./start_service.sh
```

### Option 2: Batch Job (Recommended for Long Running)

```bash
cd /home/naresh/minimax-m2-service
sbatch minimax_m2_job.sh
```

---

## 📋 Essential Commands

### Request 4 GPUs
```bash
srun --gres=gpu:4 --cpus-per-task=16 --mem=500G --time=24:00:00 --pty bash
```

### Start Service
```bash
cd /home/naresh/minimax-m2-service && ./start_service.sh
```

### Test Service
```bash
python test_client.py --test-type basic
```

### Monitor GPUs
```bash
watch -n 1 nvidia-smi
```

### Stop Service
```bash
./stop_service.sh
```

---

## 🔍 Check Status

```bash
# Check if service is running
ps aux | grep "vllm serve"

# Check port
lsof -i :8084

# Check service health
curl http://localhost:8084/health

# View logs
tail -f logs/service.log
```

---

## 📚 Full Documentation

- **SLURM_COMMANDS.md** - Complete SLURM guide with all commands
- **README.md** - Full service documentation
- **SETUP_COMPLETE.md** - Setup verification and next steps

---

**Service Port**: 8084  
**Required GPUs**: 4x H100 80GB  
**Model Size**: ~220GB



