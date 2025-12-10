#!/bin/bash
# Check if local macOS setup is ready to run pipeline

cd "/Users/leeyenshen/Desktop/Comp 545 Project"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     LOCAL SETUP CHECK - macOS                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Activate venv
source venv/bin/activate

# 1. Python environment
echo "1️⃣  Python Environment"
echo "════════════════════════════════════════════════════════════════"
python3 --version
echo "Virtual env: $(which python3)"
echo ""

# 2. Data files
echo "2️⃣  Data Files"
echo "════════════════════════════════════════════════════════════════"
if [ -f "data/raw/nq_questions.jsonl" ]; then
    echo "✅ NaturalQuestions dataset exists"
    wc -l data/raw/nq_questions.jsonl | awk '{print "   "$1" questions"}'
else
    echo "❌ NaturalQuestions dataset missing"
fi

if [ -f "data/raw/wikipedia_passages.jsonl" ]; then
    echo "✅ Wikipedia passages exist"
    wc -l data/raw/wikipedia_passages.jsonl | awk '{print "   "$1" passages"}'
else
    echo "❌ Wikipedia passages missing"
fi
echo ""

# 3. Indices
echo "3️⃣  Indices"
echo "════════════════════════════════════════════════════════════════"
if [ -d "data/indices/bm25_index" ]; then
    echo "⚠️  BM25 index exists (may not work on macOS ARM)"
else
    echo "❌ BM25 index missing (expected - Pyserini incompatible)"
fi

if [ -f "data/indices/faiss_index.bin" ]; then
    echo "✅ FAISS index exists"
    ls -lh data/indices/faiss_index.bin | awk '{print "   Size: "$5}'
else
    echo "❌ FAISS index missing"
fi
echo ""

# 4. PyTorch & MPS
echo "4️⃣  PyTorch & Apple Silicon GPU"
echo "════════════════════════════════════════════════════════════════"
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
if torch.backends.mps.is_available():
    print("✅ Apple Silicon GPU ready")
else:
    print("⚠️  MPS not available, will use CPU")
EOF
echo ""

# 5. FAISS
echo "5️⃣  FAISS"
echo "════════════════════════════════════════════════════════════════"
python3 << EOF
import faiss
print(f"FAISS installed: ✅")
print(f"FAISS GPUs detected: {faiss.get_num_gpus()}")
if faiss.get_num_gpus() == 0:
    print("⚠️  Using CPU (normal for macOS)")
EOF
echo ""

# 6. Detectors
echo "6️⃣  Hallucination Detectors"
echo "════════════════════════════════════════════════════════════════"
python3 << EOF
import sys

# RAGAS
try:
    from src.detection.ragas_detector import RAGASDetector
    print("✅ RAGAS detector available")
except Exception as e:
    print(f"❌ RAGAS detector error: {e}")

# NLI
try:
    from src.detection.nli_detector import NLIDetector
    print("✅ NLI detector available")
except Exception as e:
    print(f"❌ NLI detector error: {e}")

# Lexical
try:
    from src.detection.lexical_detector import LexicalDetector
    print("✅ Lexical detector available")
except Exception as e:
    print(f"❌ Lexical detector error: {e}")
EOF
echo ""

# 7. Config
echo "7️⃣  Configuration"
echo "════════════════════════════════════════════════════════════════"
python3 << EOF
import yaml
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f"Device: {config['generation']['device']}")
print(f"8-bit quantization: {config['generation']['load_in_8bit']}")
print(f"Model: {config['generation']['model_name']}")
EOF
echo ""

# 8. Previous results
echo "8️⃣  Previous Results"
echo "════════════════════════════════════════════════════════════════"
if [ -d "outputs/results" ]; then
    echo "Previous results found:"
    ls -lh outputs/results/*.csv 2>/dev/null | awk '{print "   "$9" ("$5")"}'
    echo ""
    echo "⚠️  Previous results exist. Re-running pipeline will overwrite them."
else
    echo "No previous results (clean state)"
fi
echo ""

# Summary
echo "════════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "════════════════════════════════════════════════════════════════"

READY=true

if [ ! -f "data/raw/nq_questions.jsonl" ]; then
    echo "❌ Missing: NaturalQuestions dataset"
    READY=false
fi

if [ ! -f "data/raw/wikipedia_passages.jsonl" ]; then
    echo "❌ Missing: Wikipedia passages"
    READY=false
fi

if [ ! -f "data/indices/faiss_index.bin" ]; then
    echo "❌ Missing: FAISS index"
    READY=false
fi

if [ "$READY" = true ]; then
    echo "✅ ALL SYSTEMS READY!"
    echo ""
    echo "To run the pipeline:"
    echo "  ./run_pipeline_safe.sh"
    echo ""
    echo "Expected time: 2-3 hours (macOS ARM with MPS)"
    echo "Note: BM25 retrieval disabled (Pyserini incompatible)"
else
    echo "⚠️  Some components missing"
    echo ""
    echo "To fix:"
    echo "  python scripts/01_download_datasets.py"
    echo "  ./run_faiss_safe.sh"
fi

echo "════════════════════════════════════════════════════════════════"
