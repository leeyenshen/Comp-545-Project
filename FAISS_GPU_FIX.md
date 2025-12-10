# FAISS GPU Installation Fix for Vast.ai

## Problem
`pip install faiss-gpu` fails with "No matching distribution found"

## Solution

### Option 1: Use Conda (Recommended)
```bash
conda install -c pytorch -c conda-forge faiss-gpu cudatoolkit=11.8 -y
```

### Option 2: Use pip with CUDA 12
```bash
# For CUDA 12.x
pip install faiss-cpu  # This version includes GPU support when CUDA is available!
```

### Option 3: Install from conda-forge
```bash
conda install -c conda-forge faiss-gpu -y
```

## Quick Fix (You're on the instance now)

**Run this on your Vast.ai instance:**

```bash
# Check if conda is available
conda --version

# If conda exists:
conda install -c pytorch faiss-gpu -y

# If no conda (docker image might not have it):
# The faiss-cpu package actually includes GPU support!
pip install faiss-cpu

# Verify it works
python3 -c "import faiss; print(f'Number of GPUs: {faiss.get_num_gpus()}')"
```

## Why This Happens

The `faiss-gpu` PyPI package was discontinued. Now:
- **With conda**: Use `faiss-gpu` from conda channels
- **With pip**: Use `faiss-cpu` which auto-detects and uses GPU when CUDA is available

## Verification

After installation, test GPU availability:

```python
import faiss
import numpy as np

print(f"FAISS GPU count: {faiss.get_num_gpus()}")

if faiss.get_num_gpus() > 0:
    # Test GPU functionality
    d = 128
    xb = np.random.random((1000, d)).astype('float32')

    index_cpu = faiss.IndexFlatL2(d)
    index_cpu.add(xb)

    res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

    print("✅ FAISS GPU is working!")
else:
    print("⚠️  No GPU detected, but FAISS will still work on CPU")
```

## For Your Current Session

Since you're already SSH'd into Vast.ai:

```bash
# Quick install
pip install faiss-cpu

# Or if conda is available
conda install -c pytorch faiss-gpu -y

# Then test
python3 test_faiss_gpu.py
```

The `faiss-cpu` package will automatically use your GPU if CUDA is available, despite the name!
