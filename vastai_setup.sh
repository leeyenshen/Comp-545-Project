#!/bin/bash
# Vast.ai Setup Script for RAG Hallucination Detection

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     RAG HALLUCINATION DETECTION - VAST.AI SETUP              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Update system
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y git wget curl vim

# Install Java (for Pyserini/BM25)
echo "☕ Installing Java for Pyserini..."
apt-get install -y openjdk-21-jdk
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Verify CUDA
echo "🔍 Checking CUDA availability..."
nvidia-smi
echo ""

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip

# Install PyTorch with CUDA support (if not already installed)
echo "🔥 Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install FAISS GPU
echo "🚀 Installing FAISS GPU..."
pip install faiss-gpu

# Install other requirements
echo "📦 Installing other dependencies..."
pip install transformers datasets sentence-transformers pyserini
pip install ragas langchain langchain-community langchain-huggingface
pip install scikit-learn pandas numpy matplotlib seaborn
pip install pyyaml nltk rouge-score bert-score
pip install accelerate bitsandbytes

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "System Information:"
echo "  CUDA: $(nvidia-smi --query-gpu=cuda_version --format=csv,noheader)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "  Java: $(java -version 2>&1 | head -n 1)"
echo ""
echo "Next steps:"
echo "1. python scripts/01_download_datasets.py"
echo "2. python scripts/02_build_bm25_index.py"
echo "3. python scripts/03_build_faiss_index.py"
echo "4. python scripts/04_run_pipeline.py"
echo "5. python scripts/05_create_visualizations.py"
