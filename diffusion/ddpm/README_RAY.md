# Ray-based Parallel Inference for DDPM

This guide explains how to use Ray to generate DDPM samples in parallel, both locally and on a KubeRay cluster.

## Overview

The `run_ray_inference.py` script loads a trained DDPM model and generates multiple samples in parallel using Ray. This is much faster than sequential generation, especially when generating hundreds or thousands of samples.

## Prerequisites

```bash
pip install ray[default] torch torchvision matplotlib numpy
```

For GPU support:
```bash
pip install ray[default] torch torchvision matplotlib numpy
```

## Usage

### 1. Local Mode (Single Machine)

Run inference on your local machine without connecting to a cluster:

```bash
python run_ray_inference.py \
  --model_path models/Unet_20241016-14.pt \
  --model_name Unet \
  --downs 8 16 32 \
  --time_emb_dim 4 \
  --img_size 28 \
  --n_samples 100 \
  --timesteps 1000 \
  --device cuda \
  --ray_address None \
  --num_gpus_per_task 0.25
```

**Key parameters:**
- `--model_path`: Path to your trained model checkpoint
- `--n_samples`: Number of samples to generate
- `--ray_address None`: Use local mode (don't connect to cluster)
- `--num_gpus_per_task 0.25`: Allocate 0.25 GPU per task (4 parallel tasks per GPU)

### 2. KubeRay Cluster Mode

#### Step 1: Deploy RayCluster

If using kind (Kubernetes in Docker) for local testing:

```bash
# Create kind cluster if needed
kind create cluster --name ray-cluster

# Install KubeRay operator (if not already installed)
kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=v1.0.0"

# Deploy the DDPM inference cluster
kubectl apply -f kuberay-config.yaml

# Check cluster status
kubectl get rayclusters
kubectl get pods
```

#### Step 2: Port Forward to Access Cluster

```bash
# Forward Ray client port
kubectl port-forward service/ddpm-inference-service 10001:10001 &

# Optional: Forward dashboard for monitoring
kubectl port-forward service/ddpm-inference-service 8265:8265 &
```

#### Step 3: Run Inference

```bash
python run_ray_inference.py \
  --model_path models/Unet_20241016-14.pt \
  --model_name Unet \
  --downs 8 16 32 \
  --time_emb_dim 4 \
  --img_size 28 \
  --n_samples 100 \
  --timesteps 1000 \
  --device cuda \
  --ray_address "auto" \
  --num_gpus_per_task 0.25
```

**Notes:**
- `--ray_address "auto"`: Automatically discover Ray cluster (works with port forwarding)
- Alternatively, use `--ray_address "ray://localhost:10001"` for explicit connection

#### Step 4: Monitor Progress

Open the Ray dashboard in your browser:
```
http://localhost:8265
```

You can see:
- Active tasks and their progress
- Resource utilization (CPU, GPU, memory)
- Task timeline and execution details

#### Step 5: Cleanup

```bash
# Delete the RayCluster
kubectl delete -f kuberay-config.yaml

# Or delete the entire kind cluster
kind delete cluster --name ray-cluster
```

## How It Works

### Parallelization Strategy

1. **Model Sharing**: The trained model weights are loaded once and put in Ray's object store
2. **Task Distribution**: Each sample generation is a separate Ray task
3. **GPU Allocation**: Each task requests a fraction of a GPU (e.g., 0.25 = 4 tasks per GPU)
4. **Result Collection**: Results are collected as tasks complete using `ray.wait()`

### Key Ray Concepts Demonstrated

- **`@ray.remote`**: Decorator to make functions executable on remote workers
- **`ray.put()`**: Store objects in distributed object store for efficient sharing
- **`ray.get()`**: Retrieve results from remote tasks
- **Resource management**: Specify GPU/CPU requirements per task
- **Cluster connection**: Connect to local or remote Ray clusters

### Performance Benefits

For generating 100 samples with T=1000 timesteps:
- **Sequential** (original code): ~100 * 30 seconds = ~50 minutes
- **Parallel with 1 GPU** (4 tasks): ~25 * 30 seconds = ~13 minutes (4x speedup)
- **Parallel with 4 GPUs** (16 tasks): ~7 * 30 seconds = ~3.5 minutes (14x speedup)

## Customizing the KubeRay Configuration

Edit `kuberay-config.yaml` to adjust:

### Worker Count
```yaml
workerGroupSpecs:
  - replicas: 3  # Increase for more parallel workers
    minReplicas: 1
    maxReplicas: 5
```

### GPU Resources
```yaml
resources:
  limits:
    nvidia.com/gpu: "1"  # GPUs per worker pod
```

### CPU/Memory
```yaml
resources:
  limits:
    cpu: "8"
    memory: "16Gi"
```

## Troubleshooting

### Connection Issues

If you can't connect to the Ray cluster:
```bash
# Check Ray head pod
kubectl get pods -l ray.io/node-type=head

# Check Ray head logs
kubectl logs -l ray.io/node-type=head

# Verify port forwarding
lsof -i :10001
```

### GPU Issues

If GPUs aren't detected:
```bash
# Check GPU availability in pods
kubectl exec -it <worker-pod-name> -- nvidia-smi

# Check Ray cluster resources
python -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"
```

### Out of Memory

If you run out of GPU memory:
- Reduce `--num_gpus_per_task` to allocate more GPU memory per task
- Generate samples in smaller batches
- Increase worker replicas to distribute load

## Learning Resources

- [Ray Documentation](https://docs.ray.io/)
- [KubeRay Documentation](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
- [Ray Core Walkthrough](https://docs.ray.io/en/latest/ray-core/walkthrough.html)
- [Ray Dashboard Guide](https://docs.ray.io/en/latest/ray-observability/getting-started.html)

## Next Steps

Once you're comfortable with this inference example, you can explore:
1. **Ray Tune** for hyperparameter optimization of your training script
2. **Ray Train** for distributed training across multiple GPUs/nodes
3. **Ray Data** for efficient data preprocessing pipelines
4. **Ray Serve** for serving your model as an API endpoint
