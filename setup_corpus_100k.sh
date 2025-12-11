#!/bin/bash
# Setup 100k diverse Wikipedia corpus for better RAG coverage

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SETUP 100K WIKIPEDIA CORPUS                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "This will:"
echo "  1. Download 100k diverse Wikipedia passages (~10-15 min)"
echo "  2. Generate embeddings with sentence-transformers (~3-5 hours)"
echo "  3. Build FAISS index (~5-10 min)"
echo ""
echo "Total estimated time: 3.5-5.5 hours"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  STEP 1/3: Download Wikipedia Passages                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
python scripts/download_wiki_dpr.py

if [ $? -ne 0 ]; then
    echo "❌ Download failed!"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  STEP 2/3: Generate Embeddings (~3-5 hours on Mac)           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "⏰ This will take 3-5 hours. You can monitor progress."
echo "   Press Ctrl+C to cancel (not recommended once started)"
echo ""
sleep 3

python scripts/03_build_faiss_index.py

if [ $? -ne 0 ]; then
    echo "❌ Embedding/indexing failed!"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SETUP COMPLETE!                                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "✓ 100k Wikipedia passages downloaded and indexed"
echo ""
echo "Next step: Run the pipeline"
echo "  ./run_pipeline_safe.sh         # All 3 quality tiers (~7-8 hours)"
echo "  ./run_pipeline_single_tier.sh  # Just HIGH tier (~2.5 hours)"
echo "  ./test_single_question.sh      # Quick test with 1 question (~5 min)"
echo ""
