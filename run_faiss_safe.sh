#!/bin/bash
# Run FAISS indexing with safe threading settings for macOS ARM

# Prevent OpenMP threading conflicts
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Prevent PyTorch from using multiple threads
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0

echo "Running FAISS indexing with safe threading settings..."
echo "OMP_NUM_THREADS=1 (to prevent crashes)"
echo ""

python scripts/03_build_faiss_index.py
