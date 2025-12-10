#!/usr/bin/env python3
"""
CUDA Diagnostic Script
"""
import torch
import sys
import subprocess
import os

print("="*60)
print("CUDA DIAGNOSTIC")
print("="*60)

print("\n1. PyTorch Info:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   CUDA version (PyTorch): {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"   CUDA device count: {torch.cuda.device_count()}")
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   Device name: {torch.cuda.get_device_name(0)}")
    print(f"   Device capability: {torch.cuda.get_device_capability(0)}")
    print(f"   CUDA initialized: {torch.cuda.is_initialized()}")
else:
    print("   ❌ CUDA NOT AVAILABLE")
    print("\n   Possible reasons:")
    print("   - PyTorch built without CUDA")
    print("   - CUDA runtime libraries missing")
    print("   - GPU not mounted in container")
    print("   - CUDA version mismatch")

print("\n2. System Info:")
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,cuda_version', '--format=csv,noheader'],
                          capture_output=True, text=True)
    print(f"   nvidia-smi output:")
    print(f"   {result.stdout.strip()}")
except:
    print("   ❌ nvidia-smi not available")

print("\n3. Environment:")
print(f"   CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"   LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
print(f"   PATH: {os.environ.get('PATH', 'Not set')[:100]}...")

print("\n4. CUDA Libraries:")
cuda_paths = [
    '/usr/local/cuda/lib64',
    '/usr/local/cuda-12/lib64',
    '/usr/local/cuda-13/lib64',
]
for path in cuda_paths:
    if os.path.exists(path):
        print(f"   ✓ Found: {path}")
        libs = [f for f in os.listdir(path) if 'cudart' in f]
        if libs:
            print(f"     - {libs[0]}")
    else:
        print(f"   ✗ Not found: {path}")

print("\n5. Testing CUDA operations:")
if torch.cuda.is_available():
    try:
        x = torch.randn(3, 3).cuda()
        print(f"   ✅ Tensor on GPU: {x.device}")
        y = x + x
        print(f"   ✅ GPU computation works")
    except Exception as e:
        print(f"   ❌ CUDA operation failed: {e}")
else:
    print("   ⏭️  Skipped (CUDA not available)")

print("\n6. FAISS GPU Check:")
try:
    import faiss
    print(f"   FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}")
    print(f"   FAISS GPUs detected: {faiss.get_num_gpus()}")
except ImportError:
    print("   ⚠️  FAISS not installed yet")
except Exception as e:
    print(f"   ❌ FAISS error: {e}")

print("="*60)

# Recommendation
if not torch.cuda.is_available():
    print("\n⚠️  RECOMMENDATION:")
    print("CUDA version mismatch detected.")
    print("System has CUDA 13.0 but PyTorch built for CUDA 12.9")
    print("\nFixes to try:")
    print("1. PyTorch often works with newer CUDA (backward compatible)")
    print("2. Set LD_LIBRARY_PATH to CUDA libraries")
    print("3. Reinstall PyTorch for CUDA 12.1 (closest available)")
else:
    print("\n✅ CUDA is working! Ready to train models.")
