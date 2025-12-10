#!/bin/bash
# Complete pipeline runner for Vast.ai with GPU acceleration

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     RAG HALLUCINATION DETECTION - FULL PIPELINE (GPU)       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "config/config.yaml" ]; then
    echo "❌ Error: config/config.yaml not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Display system info
echo "📊 System Information:"
echo "════════════════════════════════════════════════════════════════"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
echo ""
python --version
echo ""

# Test FAISS GPU
echo "🧪 Testing FAISS GPU..."
echo "════════════════════════════════════════════════════════════════"
python test_faiss_gpu.py
echo ""

# Step 1: Download datasets
echo "📥 STEP 1: Downloading Datasets"
echo "════════════════════════════════════════════════════════════════"
if [ ! -f "data/raw/natural_questions.jsonl" ]; then
    python scripts/01_download_datasets.py
    echo "✅ Datasets downloaded"
else
    echo "⏭️  Datasets already exist, skipping download"
fi
echo ""

# Step 2: Build BM25 index (sparse retrieval)
echo "🔍 STEP 2: Building BM25 Index"
echo "════════════════════════════════════════════════════════════════"
if [ ! -d "data/indices/bm25_index" ]; then
    python scripts/02_build_bm25_index.py
    echo "✅ BM25 index built"
else
    echo "⏭️  BM25 index already exists, skipping"
fi
echo ""

# Step 3: Build FAISS index (dense retrieval with GPU)
echo "🚀 STEP 3: Building FAISS Index (GPU Accelerated)"
echo "════════════════════════════════════════════════════════════════"
if [ ! -f "data/indices/faiss_index.bin" ]; then
    python scripts/03_build_faiss_index.py
    echo "✅ FAISS index built"
else
    echo "⏭️  FAISS index already exists, skipping"
fi
echo ""

# Step 4: Run detection pipeline
echo "🤖 STEP 4: Running Hallucination Detection Pipeline"
echo "════════════════════════════════════════════════════════════════"
echo "This will:"
echo "  • Generate answers with Mistral-7B (8-bit quantization)"
echo "  • Run RAGAS detection (faithfulness, relevancy, precision)"
echo "  • Run NLI detection (entailment-based)"
echo "  • Run Lexical detection (token/entity overlap)"
echo "  • Process all 3 quality tiers (high/medium/low)"
echo ""
echo "⏱️  Estimated time: 20-30 minutes with GPU"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

python scripts/04_run_pipeline.py

echo ""
echo "✅ Pipeline complete!"
echo ""

# Step 5: Generate visualizations
echo "📊 STEP 5: Creating Visualizations"
echo "════════════════════════════════════════════════════════════════"
python scripts/05_create_visualizations.py
echo "✅ Visualizations created"
echo ""

# Display results
echo "════════════════════════════════════════════════════════════════"
echo "📁 RESULTS LOCATION"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to:"
echo "  • outputs/results/*.csv       - CSV format"
echo "  • outputs/results/*.jsonl     - JSONL format"
echo "  • outputs/results/*.tex       - LaTeX tables"
echo "  • outputs/visualizations/*.png - Plots"
echo ""
echo "Result files:"
ls -lh outputs/results/
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "✅ ALL STEPS COMPLETE!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "1. Download results to your local machine:"
echo "   scp -P <PORT> root@<HOST>:/workspace/rag_hallucination/outputs/results/* ./results/"
echo ""
echo "2. Review the results:"
echo "   • Check outputs/results/results_table.csv for summary"
echo "   • View outputs/visualizations/ for plots"
echo "   • Use outputs/results/results_table.tex in your paper"
echo ""
echo "3. Fill in your LaTeX paper:"
echo "   • paper/main.tex - Add results to tables"
echo "   • paper/figures/ - Copy visualizations"
echo ""
