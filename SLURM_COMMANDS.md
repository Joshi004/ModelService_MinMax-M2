# MiniMax M2 Service - SLURM Commands & Usage Guide

This document contains all commands needed to request 4 GPUs and run the MiniMax M2 service on a SLURM cluster.

## Quick Reference

### Request 4 GPUs and Start Service

```bash
# Interactive session with 4 GPUs
srun --gres=gpu:4 --cpus-per-task=16 --mem=500G --time=24:00:00 --pty bash

# Once allocated, start the service
cd /home/naresh/minimax-m2-service
./start_service.sh
```

### Submit as Batch Job

```bash
# Submit job script
sbatch minimax_m2_job.sh

# Check job status
squeue -u $USER

# View job output
tail -f minimax_m2_job_<JOBID>.out
```

---

## Detailed Commands

### 1. Request Interactive Session with 4 GPUs

**Basic Command:**
```bash
srun --gres=gpu:4 --cpus-per-task=16 --mem=500G --time=24:00:00 --pty bash
```

**With Specific GPU Type (H100):**
```bash
srun --gres=gpu:h100:4 --cpus-per-task=16 --mem=500G --time=24:00:00 --pty bash
```

**With Partition Specification:**
```bash
srun --partition=gpu --gres=gpu:4 --cpus-per-task=16 --mem=500G --time=24:00:00 --pty bash
```

**Parameters Explained:**
- `--gres=gpu:4`: Request 4 GPUs
- `--cpus-per-task=16`: Request 16 CPU cores (4 cores per GPU)
- `--mem=500G`: Request 500GB RAM (for model loading)
- `--time=24:00:00`: Maximum runtime (24 hours)
- `--pty bash`: Interactive bash session

**Alternative: Request Specific GPU Memory:**
```bash
srun --gres=gpu:4 --gpu-mem=80G --cpus-per-task=16 --mem=500G --time=24:00:00 --pty bash
```

### 2. Start the Service (After GPU Allocation)

Once you have the interactive session with 4 GPUs:

```bash
# Navigate to service directory
cd /home/naresh/minimax-m2-service

# Start the service
./start_service.sh
```

The service will:
- Check GPU availability (should show 4 GPUs)
- Load the model across 4 GPUs
- Start vLLM server on port 8084
- Log output to `logs/service.log`

### 3. Monitor the Service

**In the same terminal (service output):**
- Service logs will be displayed in real-time
- Model loading progress will be shown
- Once ready, you'll see "Uvicorn running on http://0.0.0.0:8084"

**In another terminal (SSH to same node):**
```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Or detailed GPU info
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

# Check service process
ps aux | grep "vllm serve.*minimax"

# Check port
lsof -i :8084

# Test service health
curl http://localhost:8084/health
```

### 4. Test the Service

**Using the test client:**
```bash
# SSH to the compute node (same node where service is running)
ssh <compute-node>

# Navigate to service directory
cd /home/naresh/minimax-m2-service

# Run tests
python test_client.py --test-type basic
python test_client.py --test-type all
```

**Using curl:**
```bash
curl http://localhost:8084/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "Write a Python function to calculate factorial."
    }]
  }'
```

### 5. Stop the Service

```bash
# In the service terminal, press Ctrl+C
# Or use the stop script
cd /home/naresh/minimax-m2-service
./stop_service.sh
```

### 6. Exit SLURM Session

```bash
exit
```

---

## Batch Job Submission

### Create Job Script

Create a file `minimax_m2_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=minimax-m2
#SBATCH --output=minimax_m2_job_%j.out
#SBATCH --error=minimax_m2_job_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Optional: Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Navigate to service directory
cd /home/naresh/minimax-m2-service

# Start the service
./start_service.sh
```

### Submit the Job

```bash
# Make script executable
chmod +x minimax_m2_job.sh

# Submit job
sbatch minimax_m2_job.sh

# Check job status
squeue -u $USER

# View job output (replace JOBID with actual job ID)
tail -f minimax_m2_job_<JOBID>.out
```

### Cancel Job

```bash
# Cancel specific job
scancel <JOBID>

# Cancel all your jobs
scancel -u $USER
```

---

## SSH Tunnel for Remote Access

If you need to access the service from your local machine:

### 1. Find the Compute Node

```bash
# After job starts, find which node it's running on
squeue -u $USER

# Or check job output
grep "Node:" minimax_m2_job_<JOBID>.out
```

### 2. Create SSH Tunnel

**From your local machine:**
```bash
# Create tunnel to compute node
ssh -L 8084:<compute-node>:8084 <login-node>

# Keep this terminal open
# Now you can access the service at http://localhost:8084
```

**Example:**
```bash
# If compute node is gpu-node-01
ssh -L 8084:gpu-node-01:8084 user@login.cluster.edu
```

### 3. Test from Local Machine

```bash
# Test health endpoint
curl http://localhost:8084/health

# Test API
curl http://localhost:8084/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "Hello!"
    }]
  }'
```

---

## Complete Workflow Example

### Step-by-Step Complete Workflow

```bash
# 1. Request interactive session with 4 GPUs
srun --gres=gpu:4 --cpus-per-task=16 --mem=500G --time=24:00:00 --pty bash

# 2. Wait for allocation (you'll see a prompt on the compute node)

# 3. Verify GPUs
nvidia-smi

# 4. Navigate to service directory
cd /home/naresh/minimax-m2-service

# 5. Start the service
./start_service.sh

# 6. Wait for model loading (5-10 minutes first time, ~2-3 minutes subsequent)

# 7. In another terminal, SSH to compute node and test
ssh <compute-node>
cd /home/naresh/minimax-m2-service
python test_client.py --test-type basic

# 8. Monitor GPUs
watch -n 1 nvidia-smi

# 9. When done, stop service (Ctrl+C or ./stop_service.sh)
# 10. Exit SLURM session
exit
```

---

## Common SLURM Commands

### Check Available Resources

```bash
# Check available partitions
sinfo

# Check GPU availability
sinfo -o "%P %G %m %l %D %N" | grep gpu

# Check your job status
squeue -u $USER

# Check detailed job info
scontrol show job <JOBID>
```

### Resource Limits

```bash
# Check your account limits
sacctmgr show user $USER withassoc

# Check partition limits
scontrol show partition <partition-name>
```

### Node Information

```bash
# Show node details
scontrol show node <node-name>

# Show GPU information for nodes
scontrol show node <node-name> | grep -i gpu
```

---

## Troubleshooting

### Issue: "No GPUs available"

**Solution:**
```bash
# Check GPU availability
sinfo -o "%P %G %m %l %D %N" | grep gpu

# Try different partition
srun --partition=gpu --gres=gpu:4 --cpus-per-task=16 --mem=500G --time=24:00:00 --pty bash

# Try requesting fewer resources
srun --gres=gpu:4 --cpus-per-task=8 --mem=300G --time=12:00:00 --pty bash
```

### Issue: "Job times out"

**Solution:**
```bash
# Request longer time
srun --gres=gpu:4 --cpus-per-task=16 --mem=500G --time=48:00:00 --pty bash

# Or use batch job with longer time limit
# Edit minimax_m2_job.sh: --time=48:00:00
```

### Issue: "Out of memory"

**Solution:**
```bash
# Request more RAM
srun --gres=gpu:4 --cpus-per-task=16 --mem=600G --time=24:00:00 --pty bash
```

### Issue: "Cannot connect to service"

**Solution:**
```bash
# Check if service is running
ps aux | grep "vllm serve"

# Check if port is listening
lsof -i :8084

# Check service logs
tail -100 /home/naresh/minimax-m2-service/logs/service.log

# Verify you're on the correct node
hostname
```

### Issue: "Service starts but no GPUs detected"

**Solution:**
```bash
# Verify GPU allocation
nvidia-smi

# Check SLURM GPU allocation
echo $SLURM_GPUS_ON_NODE

# Verify CUDA_VISIBLE_DEVICES (should be set by SLURM)
echo $CUDA_VISIBLE_DEVICES

# If not set, manually set it
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## Environment Variables Set by SLURM

When SLURM allocates GPUs, it sets these variables:

```bash
# Number of GPUs allocated
echo $SLURM_GPUS_ON_NODE

# GPU IDs (comma-separated)
echo $CUDA_VISIBLE_DEVICES

# Job ID
echo $SLURM_JOB_ID

# Node name
echo $SLURM_NODELIST

# Number of CPUs
echo $SLURM_CPUS_PER_TASK
```

---

## Recommended Resource Requests

### For MiniMax M2 (4x H100 80GB)

**Minimum:**
```bash
--gres=gpu:4 --cpus-per-task=16 --mem=400G --time=24:00:00
```

**Recommended:**
```bash
--gres=gpu:4 --cpus-per-task=16 --mem=500G --time=24:00:00
```

**For Long Running:**
```bash
--gres=gpu:4 --cpus-per-task=16 --mem=500G --time=72:00:00
```

**For Development/Testing:**
```bash
--gres=gpu:4 --cpus-per-task=8 --mem=300G --time=4:00:00
```

---

## Quick Command Reference Card

```bash
# ============================================
# REQUEST GPUs
# ============================================
srun --gres=gpu:4 --cpus-per-task=16 --mem=500G --time=24:00:00 --pty bash

# ============================================
# START SERVICE
# ============================================
cd /home/naresh/minimax-m2-service && ./start_service.sh

# ============================================
# MONITOR
# ============================================
watch -n 1 nvidia-smi
tail -f /home/naresh/minimax-m2-service/logs/service.log

# ============================================
# TEST
# ============================================
python test_client.py --test-type basic
curl http://localhost:8084/health

# ============================================
# STOP
# ============================================
./stop_service.sh
# or Ctrl+C

# ============================================
# EXIT
# ============================================
exit
```

---

## Notes

1. **First Run**: Model download (~220GB) happens automatically on first start
2. **Model Loading**: Takes 5-10 minutes first time, 2-3 minutes subsequent runs
3. **Port**: Service runs on port 8084 (ensure it's not in use)
4. **GPUs**: Requires exactly 4 GPUs (H100 80GB recommended)
5. **Memory**: Model uses ~220GB GPU memory across 4 GPUs
6. **Time Limit**: Set appropriate time limit based on your needs
7. **SSH Tunnel**: Required if accessing from outside the cluster

---

**Last Updated**: Setup completion date
**Service Location**: `/home/naresh/minimax-m2-service/`
**Service Port**: 8084
**Required GPUs**: 4x H100 80GB

