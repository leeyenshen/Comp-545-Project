#!/bin/bash
# OPTIMIZED pipeline runner for macOS with parallel processing

set -e  # Exit on error

cd "/Users/leeyenshen/Desktop/Comp 545 Project"

# Optimize threading for macOS (prevent crashes but allow parallelism)
export OMP_NUM_THREADS=2  # Allow 2 threads (instead of 1)
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

# Python optimization
export PYTHONOPTIMIZE=1

# MPS optimization for Apple Silicon
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Aggressive memory management

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     RAG HALLUCINATION DETECTION - FAST PIPELINE             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  • Using parallel detector execution (3x speedup)"
echo "  • OMP threads: 2 (balanced for stability)"
echo "  • Apple Silicon MPS acceleration"
echo "  • FAISS (dense) retrieval only"
echo ""
echo "Expected time: 1-1.5 hours (vs 2-3 hours normal)"
echo ""

# Activate virtual environment
source venv/bin/activate

# Run optimized pipeline
python3 scripts/04_run_pipeline_parallel.py

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Pipeline complete! Results saved to outputs/results/"
echo "════════════════════════════════════════════════════════════════"
