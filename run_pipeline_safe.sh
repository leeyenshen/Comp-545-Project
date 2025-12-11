#!/bin/bash
# Run the full pipeline with safe settings for macOS ARM

# Prevent OpenMP threading conflicts
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     RAG HALLUCINATION DETECTION - FULL PIPELINE              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  • Using FAISS (dense) retrieval only"
echo "  • BM25 disabled (Pyserini incompatible with macOS ARM)"
echo "  • Safe threading settings enabled"
echo ""

# Activate virtual environment
source venv/bin/activate

python scripts/04_run_pipeline.py

echo ""
echo "Pipeline complete! Results saved to outputs/results/"
