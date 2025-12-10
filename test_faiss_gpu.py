#!/usr/bin/env python3
"""
Test FAISS GPU availability and performance
"""

import numpy as np
import time

print("="*60)
print("FAISS GPU TEST")
print("="*60)

# Test 1: Import FAISS
print("\n1. Testing FAISS import...")
try:
    import faiss
    print("   ✅ FAISS imported successfully")
except ImportError as e:
    print(f"   ❌ FAISS import failed: {e}")
    print("   Fix: pip install faiss-gpu (or faiss-cpu)")
    exit(1)

# Test 2: Check GPU availability
print("\n2. Checking GPU availability...")
num_gpus = faiss.get_num_gpus()
print(f"   Number of GPUs detected: {num_gpus}")

if num_gpus == 0:
    print("   ⚠️  No GPUs available - FAISS will use CPU")
    print("   This is normal on macOS or systems without CUDA")
else:
    print(f"   ✅ {num_gpus} GPU(s) available for FAISS")

# Test 3: Basic FAISS functionality
print("\n3. Testing basic FAISS functionality...")
d = 128  # dimension
n = 10000  # number of vectors
xb = np.random.random((n, d)).astype('float32')

# Build CPU index
index_cpu = faiss.IndexFlatL2(d)
index_cpu.add(xb)
print(f"   ✅ Created CPU index with {index_cpu.ntotal} vectors")

# Test 4: GPU performance (if available)
if num_gpus > 0:
    print("\n4. Testing GPU performance...")

    try:
        # Convert to GPU
        res = faiss.StandardGpuResources()
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        print("   ✅ Successfully converted index to GPU")

        # Create test queries
        xq = np.random.random((100, d)).astype('float32')
        k = 10

        # Benchmark CPU
        start = time.time()
        D_cpu, I_cpu = index_cpu.search(xq, k)
        cpu_time = time.time() - start

        # Benchmark GPU
        start = time.time()
        D_gpu, I_gpu = index_gpu.search(xq, k)
        gpu_time = time.time() - start

        # Compare
        print(f"\n   Performance Comparison (100 queries, k={k}):")
        print(f"   CPU time: {cpu_time*1000:.2f} ms")
        print(f"   GPU time: {gpu_time*1000:.2f} ms")
        print(f"   Speedup: {cpu_time/gpu_time:.2f}x faster on GPU")

        # Verify results match
        if np.allclose(D_cpu, D_gpu) and np.allclose(I_cpu, I_gpu):
            print("   ✅ GPU results match CPU results")
        else:
            print("   ⚠️  GPU results differ slightly from CPU (expected due to floating point)")

    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
else:
    print("\n4. GPU performance test skipped (no GPU available)")

# Test 5: Memory info (if GPU available)
if num_gpus > 0:
    print("\n5. GPU memory info...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"   Available VRAM: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
    except:
        print("   (PyTorch not available for detailed GPU info)")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)

if num_gpus > 0:
    print("\n✅ Your system is ready for GPU-accelerated FAISS!")
else:
    print("\n⚠️  FAISS will run on CPU (slower but functional)")
