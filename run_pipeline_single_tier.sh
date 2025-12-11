#!/bin/bash
# Run pipeline for a single quality tier (for testing)

# Prevent OpenMP threading conflicts
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     RAG HALLUCINATION DETECTION - SINGLE TIER TEST           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Activate virtual environment
source venv/bin/activate

# Run for HIGH tier only using importlib
python3 << 'EOF'
import sys
import importlib.util
from pathlib import Path

# Load the pipeline module
spec = importlib.util.spec_from_file_location("pipeline", "scripts/04_run_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
sys.modules["pipeline"] = pipeline
spec.loader.exec_module(pipeline)

# Run for HIGH tier only
config = pipeline.load_config()
print("Running HIGH tier only...")
results = pipeline.run_full_pipeline(config, 'high', num_questions=50)
print(f"\n✓ Completed! Generated {len(results)} results")
EOF

echo ""
echo "Single tier test complete!"
