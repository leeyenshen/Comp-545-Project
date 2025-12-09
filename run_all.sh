#!/bin/bash

# Master script to run the complete RAG hallucination detection pipeline
# Usage: ./run_all.sh [--skip-data] [--skip-indices] [--quick-test]

set -e  # Exit on error

echo "======================================================================"
echo "RAG HALLUCINATION DETECTION - COMPLETE PIPELINE"
echo "======================================================================"
echo ""

# Parse arguments
SKIP_DATA=false
SKIP_INDICES=false
QUICK_TEST=false

for arg in "$@"; do
    case $arg in
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-indices)
            SKIP_INDICES=true
            shift
            ;;
        --quick-test)
            QUICK_TEST=true
            shift
            ;;
    esac
done

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
echo ""

# Week 1: Data and Retrieval Setup
echo "======================================================================"
echo "WEEK 1: DATA AND RETRIEVAL SETUP"
echo "======================================================================"
echo ""

if [ "$SKIP_DATA" = false ]; then
    echo "Step 1: Downloading datasets..."
    python scripts/01_download_datasets.py
    echo ""
else
    echo "Step 1: Skipping data download (--skip-data)"
    echo ""
fi

if [ "$SKIP_INDICES" = false ]; then
    echo "Step 2: Building BM25 index..."
    python scripts/02_build_bm25_index.py
    echo ""

    echo "Step 3: Building FAISS index..."
    python scripts/03_build_faiss_index.py
    echo ""
else
    echo "Steps 2-3: Skipping index building (--skip-indices)"
    echo ""
fi

# Week 2-3: Pipeline and Evaluation
echo "======================================================================"
echo "WEEKS 2-3: PIPELINE EXECUTION AND EVALUATION"
echo "======================================================================"
echo ""

if [ "$QUICK_TEST" = true ]; then
    echo "Running QUICK TEST mode (fewer samples)..."
    # Modify config temporarily for quick test
    # This would require a config override mechanism
    echo "Note: Edit scripts/04_run_pipeline.py to set num_questions=10 for quick test"
fi

echo "Step 4: Running full RAG pipeline..."
echo "This will:"
echo "  - Retrieve contexts at 3 quality tiers"
echo "  - Generate answers with LLM"
echo "  - Run hallucination detectors"
echo "  - Save results"
echo ""
echo "⚠️  This step may take 1-4 hours depending on your GPU"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/04_run_pipeline.py
    echo ""
else
    echo "Skipping pipeline execution."
    echo ""
fi

# Week 3: Evaluation and Visualization
echo "======================================================================"
echo "WEEK 3: EVALUATION AND VISUALIZATION"
echo "======================================================================"
echo ""

echo "Step 5: Running evaluation..."
python src/evaluation/evaluator.py
echo ""

echo "Step 6: Creating visualizations..."
python scripts/05_create_visualizations.py
echo ""

# Summary
echo "======================================================================"
echo "PIPELINE COMPLETE!"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  - outputs/results/         (JSON and CSV results)"
echo "  - outputs/visualizations/  (Plots and figures)"
echo ""
echo "Next steps:"
echo "  1. Review results in outputs/results/"
echo "  2. Check visualizations in outputs/visualizations/"
echo "  3. Compile paper in paper/main.tex"
echo ""
echo "To compile the paper (requires LaTeX):"
echo "  cd paper"
echo "  pdflatex main.tex"
echo "  bibtex main"
echo "  pdflatex main.tex"
echo "  pdflatex main.tex"
echo ""
echo "======================================================================"
