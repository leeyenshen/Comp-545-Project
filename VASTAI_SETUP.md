# Vast.ai Setup Guide - RAG Hallucination Detection with FAISS GPU

## Step 1: Choose Vast.ai Instance

### Recommended Specifications:
- **GPU**: RTX 3060/3070/4070 or better (12GB+ VRAM recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ disk space
- **Docker Image**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` or similar
- **Cost**: ~$0.20-0.40/hour

### Search Filters on Vast.ai:
```
GPU RAM >= 12 GB
System RAM >= 16 GB
Disk Space >= 50 GB
CUDA Version >= 11.8
```

---

## Step 2: Create Setup Script

Save this as `vastai_setup.sh` in your project:

```bash
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

# Create project directory
echo "📁 Setting up project directory..."
cd /workspace
mkdir -p rag_hallucination

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip

# Install PyTorch with CUDA support (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install FAISS GPU
echo "🚀 Installing FAISS GPU..."
pip install faiss-gpu

# Install other requirements
pip install transformers datasets sentence-transformers pyserini
pip install ragas langchain langchain-community langchain-huggingface
pip install scikit-learn pandas numpy matplotlib seaborn
pip install pyyaml nltk rouge-score bert-score

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Upload your project files"
echo "2. Run: python scripts/01_download_datasets.py"
echo "3. Run: python scripts/02_build_bm25_index.py"
echo "4. Run: python scripts/03_build_faiss_index.py"
echo "5. Run: python scripts/04_run_pipeline.py"
```

---

## Step 3: Transfer Project to Vast.ai

### Option A: Using SCP (Secure Copy)
```bash
# From your local machine
# First, get Vast.ai SSH connection details from the instance page

# Compress project (excluding venv and large files)
cd "/Users/leeyenshen/Desktop/Comp 545 Project"
tar --exclude='venv' --exclude='data' --exclude='outputs' --exclude='*.pyc' \
    -czf rag_project.tar.gz .

# Copy to Vast.ai
scp -P <PORT> rag_project.tar.gz root@<HOST>:/workspace/

# SSH into Vast.ai
ssh -p <PORT> root@<HOST>

# Extract on Vast.ai
cd /workspace
tar -xzf rag_project.tar.gz -C rag_hallucination/
cd rag_hallucination
```

### Option B: Using Git (Recommended)
```bash
# On Vast.ai instance
cd /workspace
git clone <YOUR_REPO_URL> rag_hallucination
cd rag_hallucination
```

---

## Step 4: Run Setup Script

```bash
# On Vast.ai instance
cd /workspace/rag_hallucination
chmod +x vastai_setup.sh
./vastai_setup.sh
```

---

## Step 5: Verify FAISS GPU Installation

Create and run this test script:

```python
# test_faiss_gpu.py
import faiss
import numpy as np

print("Testing FAISS GPU...")

# Check if GPU is available
print(f"Number of GPUs available: {faiss.get_num_gpus()}")

if faiss.get_num_gpus() > 0:
    # Create test data
    d = 128  # dimension
    n = 10000  # number of vectors
    xb = np.random.random((n, d)).astype('float32')

    # Build CPU index
    index_cpu = faiss.IndexFlatL2(d)
    index_cpu.add(xb)

    # Convert to GPU
    res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

    # Test search
    xq = np.random.random((5, d)).astype('float32')
    k = 4
    D, I = index_gpu.search(xq, k)

    print("✅ FAISS GPU is working!")
    print(f"Search results shape: {I.shape}")
else:
    print("❌ No GPU detected for FAISS")
```

Run it:
```bash
python test_faiss_gpu.py
```

---

## Step 6: Update Configuration for GPU

Your config is already updated! Verify [config/config.yaml](config/config.yaml):
```yaml
generation:
  device: "auto"  # ✅ Already set
  load_in_8bit: true  # ✅ Already set
```

---

## Step 7: Modify FAISS Indexing for GPU

Update `scripts/03_build_faiss_index.py` to use GPU:

```python
# Add this after building the CPU index (around line 50-60)

import faiss

# Build CPU index first
index_cpu = faiss.IndexFlatL2(embeddings.shape[1])
index_cpu.add(embeddings)

# Convert to GPU if available
if faiss.get_num_gpus() > 0:
    print("🚀 Converting FAISS index to GPU...")
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    print("✅ FAISS index running on GPU")
else:
    print("⚠️  No GPU available, using CPU index")
    index = index_cpu

# Save CPU version for portability
faiss.write_index(index_cpu, str(index_path))
```

---

## Step 8: Run Full Pipeline

```bash
# On Vast.ai instance
cd /workspace/rag_hallucination

# 1. Download datasets
echo "📥 Downloading datasets..."
python scripts/01_download_datasets.py

# 2. Build BM25 index (should work on Linux!)
echo "🔍 Building BM25 index..."
python scripts/02_build_bm25_index.py

# 3. Build FAISS index (GPU accelerated)
echo "🚀 Building FAISS GPU index..."
python scripts/03_build_faiss_index.py

# 4. Run detection pipeline
echo "🤖 Running hallucination detection pipeline..."
python scripts/04_run_pipeline.py

# 5. Generate visualizations
echo "📊 Creating visualizations..."
python scripts/05_create_visualizations.py

echo "✅ Pipeline complete!"
```

---

## Step 9: Download Results

```bash
# From your local machine
# Compress results on Vast.ai
ssh -p <PORT> root@<HOST> "cd /workspace/rag_hallucination && tar -czf results.tar.gz outputs/"

# Download to local
scp -P <PORT> root@<HOST>:/workspace/rag_hallucination/results.tar.gz .

# Extract locally
tar -xzf results.tar.gz
```

---

## Step 10: Monitor GPU Usage

```bash
# On Vast.ai instance
watch -n 1 nvidia-smi
```

---

## Expected Performance Improvements

| Component | CPU (macOS) | GPU (Vast.ai) | Speedup |
|-----------|-------------|---------------|---------|
| FAISS Indexing | ~5-10 min | ~1-2 min | **5x faster** |
| FAISS Search | ~2-3 sec/query | ~0.1-0.2 sec/query | **15x faster** |
| Mistral-7B Inference | ~5-10 sec/answer | ~1-2 sec/answer | **5x faster** |
| 8-bit Quantization | ❌ Not available | ✅ Available | **2x faster** |
| Overall Pipeline | ~2-3 hours | ~20-30 min | **4-6x faster** |

---

## Troubleshooting

### Issue: FAISS GPU not detected
```bash
# Check CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Reinstall faiss-gpu
pip uninstall faiss-cpu faiss-gpu
pip install faiss-gpu
```

### Issue: Out of GPU memory
```python
# In config.yaml, reduce batch size or model size
generation:
  max_new_tokens: 100  # Reduce from 200
  load_in_8bit: true   # Enable quantization
```

### Issue: Pyserini still failing
```bash
# Verify Java installation
java -version
echo $JAVA_HOME

# Should show Java 21
# If not, reinstall
apt-get install -y openjdk-21-jdk
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
```

---

## Cost Estimation

**For full pipeline run (all 3 quality tiers):**
- Instance cost: $0.30/hour × 0.5 hours = **$0.15**
- Storage: Minimal (included)
- **Total: ~$0.15-0.30** for complete experiment

**vs Local macOS:**
- Time: 2-3 hours → **30 minutes on GPU**
- No crashes, no workarounds needed
- BM25 works natively

---

## Quick Reference Commands

```bash
# SSH into Vast.ai
ssh -p <PORT> root@<HOST>

# Check GPU
nvidia-smi

# Activate environment (if needed)
cd /workspace/rag_hallucination

# Run pipeline
python scripts/04_run_pipeline.py

# Monitor in real-time
tail -f outputs/results/*.jsonl

# Download results
# (from local machine)
scp -P <PORT> root@<HOST>:/workspace/rag_hallucination/outputs/results/* ./results/
```

---

## Next Steps After Setup

1. ✅ Verify FAISS GPU working with test script
2. ✅ Run data download
3. ✅ Build indices (BM25 + FAISS GPU)
4. ✅ Run full pipeline (all 3 quality tiers)
5. ✅ Generate visualizations
6. ✅ Download results to local machine
7. ✅ Fill in LaTeX paper with results

**Estimated total time: 30-45 minutes** (vs 2-3 hours on macOS)
