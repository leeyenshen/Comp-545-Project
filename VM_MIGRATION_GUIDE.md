# VM Migration Guide

## Summary
✅ **YES, this code will work on a Linux VM** (likely better than macOS ARM!)

The code is mostly platform-independent. Only a few configuration changes are needed.

---

## Required Changes for Linux VM

### 1. **Config File Changes** (`config/config.yaml`)

**Lines 40-41: Device Settings**
```yaml
# BEFORE (macOS):
device: "mps"  # auto, cuda, cpu, mps (for Apple Silicon)
load_in_8bit: false  # Disabled for macOS (requires CUDA)

# AFTER (Linux VM with GPU):
device: "auto"  # Will auto-detect CUDA
load_in_8bit: true  # Enable for faster inference (requires CUDA)

# OR (Linux VM without GPU):
device: "cpu"
load_in_8bit: false
```

### 2. **Shell Scripts** (Optional - these are macOS workarounds)

**Files to DELETE or IGNORE on VM:**
- `fix_java_env.sh` - macOS Homebrew-specific paths
- `run_bm25_with_java.sh` - macOS Java workaround
- `run_faiss_safe.sh` - macOS OpenMP threading workaround
- `run_pipeline_safe.sh` - macOS OpenMP threading workaround

**On Linux VM, use directly:**
```bash
# Instead of ./run_pipeline_safe.sh, just run:
python scripts/04_run_pipeline.py
```

### 3. **Python Requirements** (`requirements.txt`)

**No changes needed** - all dependencies are cross-platform.

However, on Linux you may want to:
- Install `faiss-gpu` instead of `faiss-cpu` (if you have CUDA)
- Pyserini should work natively (no Java issues like macOS ARM)

---

## Benefits of Running on Linux VM

### ✅ Advantages:
1. **Pyserini/BM25 will work** - No Java/JVM path issues
2. **No OpenMP threading crashes** - Can use multiple threads safely
3. **8-bit quantization available** - If VM has GPU (faster inference)
4. **CUDA acceleration** - Much faster than macOS MPS
5. **No SSL certificate issues** - NLTK downloads work smoothly
6. **Better PyTorch support** - Linux is PyTorch's primary platform

### ⚠️ Disadvantages:
- None really! Linux is the recommended platform for this kind of work

---

## What Will Work Without Changes

✅ **All Python code** - Uses `torch.cuda.is_available()` checks
✅ **All data scripts** - Platform-independent file operations
✅ **All detector modules** - No platform-specific code
✅ **RAGAS with local models** - Already configured
✅ **NLI detector** - Works perfectly
✅ **Lexical detector** - Works perfectly
✅ **FAISS indexing** - Cross-platform
✅ **Result generation** - Cross-platform

---

## Migration Checklist

```bash
# 1. Copy project to VM (exclude venv)
rsync -av --exclude='venv' /path/to/project/ vm:/path/to/project/

# 2. On VM, create new venv
cd /path/to/project
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# Optional: Install GPU version of FAISS
pip uninstall faiss-cpu
pip install faiss-gpu

# 4. Edit config/config.yaml
# Change device: "mps" to device: "auto"
# Change load_in_8bit: false to load_in_8bit: true (if GPU available)

# 5. Download data
python scripts/01_download_datasets.py

# 6. Build indices
python scripts/02_build_bm25_index.py  # Should work on Linux!
python scripts/03_build_faiss_index.py

# 7. Run pipeline
python scripts/04_run_pipeline.py  # No need for threading workarounds!

# 8. Generate visualizations
python scripts/05_create_visualizations.py
```

---

## Platform-Specific Issues Summary

| Issue | macOS ARM | Linux VM |
|-------|-----------|----------|
| Pyserini/BM25 | ❌ Java issues | ✅ Works |
| OpenMP Threading | ❌ Crashes | ✅ Works |
| 8-bit Quantization | ❌ No CUDA | ✅ Works (if GPU) |
| CUDA Acceleration | ❌ Only MPS | ✅ Full CUDA |
| Threading Env Vars | ⚠️ Required | ✅ Not needed |
| NLTK SSL | ⚠️ Certificate issues | ✅ Works |
| PyTorch Support | ⚠️ Secondary platform | ✅ Primary platform |

---

## Conclusion

**YES, migrate to Linux VM!** It will solve all your macOS-specific issues:
- No more OpenMP crashes
- No more Java/Pyserini issues
- Faster inference with CUDA
- Native PyTorch support

Only change needed: **Edit 2 lines in `config/config.yaml`**
