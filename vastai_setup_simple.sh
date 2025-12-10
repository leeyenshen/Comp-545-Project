#!/bin/bash
# Simplified Vast.ai Setup Script - Works with pip only

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     RAG HALLUCINATION DETECTION - VAST.AI SETUP              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check CUDA
echo "🔍 Checking CUDA availability..."
nvidia-smi
echo ""

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✓ Conda detected"
    USE_CONDA=true
else
    echo "⚠️  Conda not found, using pip only"
    USE_CONDA=false
fi

# Update pip
echo "📦 Updating pip..."
pip install --upgrade pip

# Install Java (for Pyserini/BM25)
echo "☕ Installing Java for Pyserini..."
apt-get update -qq
apt-get install -y openjdk-21-jdk
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
echo "export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64" >> ~/.bashrc
echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> ~/.bashrc

# Install PyTorch with CUDA (if not already installed)
echo "🔥 Installing/Verifying PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install FAISS with GPU support
echo "🚀 Installing FAISS with GPU support..."
if [ "$USE_CONDA" = true ]; then
    echo "Using conda for FAISS..."
    conda install -c pytorch -c conda-forge faiss-gpu cudatoolkit=11.8 -y
else
    echo "Using pip for FAISS..."
    # Try different package names
    pip install faiss-gpu || pip install faiss-cpu  # faiss-cpu often has GPU support
fi

# Verify FAISS GPU
echo ""
echo "Verifying FAISS GPU..."
python3 -c "import faiss; print(f'FAISS GPU count: {faiss.get_num_gpus()}')" || echo "⚠️  FAISS installed but GPU check failed"

# Install other dependencies
echo ""
echo "📦 Installing other dependencies..."
pip install transformers datasets sentence-transformers pyserini
pip install ragas langchain langchain-community langchain-huggingface
pip install scikit-learn pandas numpy matplotlib seaborn
pip install pyyaml nltk rouge-score bert-score
pip install accelerate bitsandbytes

# Download NLTK data
echo ""
echo "📚 Downloading NLTK data..."
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "System Information:"
nvidia-smi --query-gpu=name,memory.total,cuda_version --format=csv,noheader
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"
python3 -c "import faiss; print(f'FAISS GPUs: {faiss.get_num_gpus()}')"
echo ""
echo "Next steps:"
echo "1. python scripts/01_download_datasets.py"
echo "2. python scripts/02_build_bm25_index.py"
echo "3. python scripts/03_build_faiss_index.py"
echo "4. python scripts/04_run_pipeline.py"
