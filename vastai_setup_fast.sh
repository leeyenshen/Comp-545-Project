#!/bin/bash
# FAST Vast.ai Setup Script with optimizations

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     RAG HALLUCINATION DETECTION - FAST SETUP                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Optimization 1: Use multiple parallel downloads
export PIP_DOWNLOAD_CACHE=/tmp/pip-cache
export PIP_NO_CACHE_DIR=0

# Optimization 2: Disable unnecessary features
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_WARN_SCRIPT_LOCATION=1

# Update system packages (minimal)
echo "📦 Updating essential packages..."
apt-get update -qq
apt-get install -y openjdk-21-jdk git wget curl vim -qq

# Set up Java for Pyserini
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
echo "export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64" >> ~/.bashrc

# Verify CUDA
echo "🔍 GPU Check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Optimization 3: Upgrade pip with --no-warn-script-location
echo "⚡ Upgrading pip..."
pip install --upgrade pip -q

# Optimization 4: Install packages in batches with --no-deps where safe
echo "🚀 Installing core ML packages (this is the slow part)..."
echo "   Installing PyTorch..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "   Installing transformers stack..."
pip install -q transformers accelerate bitsandbytes

echo "   Installing FAISS..."
if command -v conda &> /dev/null; then
    conda install -c pytorch faiss-gpu -y -q
else
    pip install -q faiss-cpu
fi

echo "   Installing sentence-transformers and datasets..."
pip install -q sentence-transformers datasets

echo "   Installing Pyserini..."
pip install -q pyserini

echo "   Installing RAGAS and LangChain..."
pip install -q ragas langchain langchain-community langchain-huggingface

echo "   Installing scientific packages..."
pip install -q scikit-learn pandas numpy matplotlib seaborn

echo "   Installing utilities..."
pip install -q pyyaml nltk rouge-score bert-score

# Download NLTK data
echo "📚 Downloading NLTK data..."
python3 << EOF
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("✓ NLTK data downloaded")
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "System Check:"
python3 -c "import torch; print(f'  PyTorch CUDA: {torch.cuda.is_available()}')"
python3 -c "import faiss; print(f'  FAISS GPUs: {faiss.get_num_gpus()}')"
java -version 2>&1 | head -n 1 | sed 's/^/  /'
echo ""
echo "Ready to run pipeline!"
