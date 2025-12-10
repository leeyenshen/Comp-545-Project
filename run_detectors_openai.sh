#!/bin/bash
# Re-run detectors with OpenAI API for RAGAS (FAST!)

set -e

cd "/Users/leeyenshen/Desktop/Comp 545 Project"

# Load OpenAI API key from .env if exists
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo ""
    echo "Please run setup first:"
    echo "  ./setup_openai.sh"
    echo ""
    echo "Or set it manually:"
    echo "  export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Safe threading for macOS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     RE-RUN DETECTORS WITH OPENAI (FAST!)                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  • RAGAS: OpenAI API (GPT-3.5-turbo)"
echo "  • NLI: Local RoBERTa model"
echo "  • Lexical: Local overlap detector"
echo ""
echo "⏱️  Expected time: 5-10 minutes"
echo "💰 Estimated cost: ~\$0.10-0.50"
echo ""

# Activate venv
source venv/bin/activate

# Run detectors with OpenAI flag
python3 scripts/rerun_detectors_openai.py

echo ""
echo "✅ Detectors complete!"
echo ""
echo "Next steps:"
echo "  1. Check updated results in outputs/results/"
echo "  2. Run: python scripts/05_create_visualizations.py"
