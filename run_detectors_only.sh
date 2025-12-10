#!/bin/bash
# Re-run only detectors on existing results (FAST!)

set -e

cd "/Users/leeyenshen/Desktop/Comp 545 Project"

# Safe threading for macOS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     RE-RUN DETECTORS ONLY (FAST MODE)                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "This will:"
echo "  • Load existing QA results from outputs/results/"
echo "  • Run RAGAS, NLI, and Lexical detectors"
echo "  • Update results with detector scores"
echo ""
echo "⏱️  Expected time: 15-30 minutes (vs 2-3 hours for full pipeline)"
echo ""

# Activate venv
source venv/bin/activate

# Run detectors
python3 scripts/rerun_detectors_only.py

echo ""
echo "✅ Detectors complete!"
echo ""
echo "Next steps:"
echo "  1. Check updated results in outputs/results/"
echo "  2. Run: python scripts/05_create_visualizations.py"
