# Vast.ai Quick Start Guide

## 🚀 Super Fast Setup (5 minutes)

### 1. Choose Instance on Vast.ai
- Go to [vast.ai](https://vast.ai)
- Search for: `RTX 3060` or better
- Filter: `>= 12 GB VRAM`, `>= 16 GB RAM`
- Pick cheapest (~$0.20-0.40/hour)
- Select Docker: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`

### 2. SSH into Instance
```bash
# From Vast.ai instance page, copy SSH command
ssh -p <PORT> root@<HOST>
```

### 3. Upload Project
**Option A: Git (Recommended)**
```bash
cd /workspace
git clone https://github.com/<your-repo>.git rag_hallucination
cd rag_hallucination
```

**Option B: SCP from Local**
```bash
# On local machine
cd "/Users/leeyenshen/Desktop/Comp 545 Project"
tar --exclude='venv' --exclude='data' --exclude='outputs' -czf project.tar.gz .
scp -P <PORT> project.tar.gz root@<HOST>:/workspace/

# On Vast.ai
cd /workspace
mkdir rag_hallucination
tar -xzf project.tar.gz -C rag_hallucination/
cd rag_hallucination
```

### 4. Run Setup
```bash
chmod +x vastai_setup.sh
./vastai_setup.sh
```
⏱️ Takes ~5 minutes

### 5. Test GPU
```bash
python test_faiss_gpu.py
```
Should see: `✅ GPU(s) available for FAISS`

### 6. Run Full Pipeline
```bash
chmod +x run_vastai_pipeline.sh
./run_vastai_pipeline.sh
```
⏱️ Takes ~20-30 minutes

### 7. Download Results
```bash
# From local machine
scp -P <PORT> -r root@<HOST>:/workspace/rag_hallucination/outputs/ ./
```

---

## 📋 One-Line Commands

### Quick Run (After Setup)
```bash
cd /workspace/rag_hallucination && python scripts/04_run_pipeline.py
```

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Check Progress
```bash
tail -f outputs/results/*.jsonl
```

### Download Just Results
```bash
# From local machine
scp -P <PORT> root@<HOST>:/workspace/rag_hallucination/outputs/results/*.csv ./
```

---

## 🎯 Expected Performance

| Task | Time on GPU | Time on macOS |
|------|-------------|---------------|
| Dataset Download | ~2 min | ~2 min |
| BM25 Index | ~3 min | ❌ Failed |
| FAISS Index | ~2 min | ~10 min |
| Pipeline (3 tiers) | ~25 min | ~3 hours |
| **TOTAL** | **~30 min** | **~3+ hours** |

**Cost: ~$0.20** (0.5 hours × $0.40/hour)

---

## ✅ What You Get

After pipeline completes:

```
outputs/
├── results/
│   ├── results_high.csv      # High quality (80% relevant)
│   ├── results_medium.csv    # Medium quality (50% relevant)
│   ├── results_low.csv       # Low quality (20% relevant)
│   ├── results_table.csv     # Summary table
│   └── results_table.tex     # LaTeX table for paper
└── visualizations/
    ├── performance_vs_quality.png
    ├── confusion_matrices.png
    ├── auroc_heatmap.png
    └── precision_recall_curves.png
```

---

## 🐛 Troubleshooting

### "No GPU detected"
```bash
nvidia-smi  # Should show your GPU
python -c "import torch; print(torch.cuda.is_available())"  # Should be True
```

### "FAISS GPU not working"
```bash
pip uninstall faiss-cpu faiss-gpu
pip install faiss-gpu
python test_faiss_gpu.py
```

### "Out of memory"
Edit `config/config.yaml`:
```yaml
generation:
  max_new_tokens: 100  # Reduce from 200
```

### "Pyserini/BM25 failing"
```bash
# Check Java
java -version  # Should show Java 21
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
python scripts/02_build_bm25_index.py
```

---

## 💰 Cost Optimization

### Minimize Cost:
1. **Use spot instances** (much cheaper but can be interrupted)
2. **Choose older GPUs** (RTX 3060 vs 4090)
3. **Run during off-peak hours** (prices vary)
4. **Destroy instance when done** (don't forget!)

### Save Progress:
```bash
# Before destroying instance, backup everything
cd /workspace/rag_hallucination
tar -czf backup.tar.gz .
# Download backup.tar.gz to local machine
```

---

## 🎓 Pro Tips

### 1. Screen for Long Runs
```bash
# Start screen session
screen -S rag_pipeline

# Run pipeline
./run_vastai_pipeline.sh

# Detach: Ctrl+A, then D
# Reattach later: screen -r rag_pipeline
```

### 2. Monitor from Local Machine
```bash
# SSH with port forwarding
ssh -p <PORT> -L 8888:localhost:8888 root@<HOST>

# Start Jupyter on Vast.ai
jupyter notebook --no-browser --port=8888

# Open in local browser: http://localhost:8888
```

### 3. Auto-Download Results
```bash
# Add to run_vastai_pipeline.sh at the end:
tar -czf results_$(date +%Y%m%d_%H%M%S).tar.gz outputs/
```

---

## 📊 Verify Results

### Check if all detectors worked:
```bash
# Should see filled columns for ragas, nli, and lexical
head -2 outputs/results/results_high.csv
```

Expected columns:
```
question,ground_truth,answer,quality_tier,num_relevant,num_distractors,
ragas_hallucinated,ragas_faithfulness,
nli_hallucinated,nli_entailment_prob,
lexical_hallucinated,lexical_overlap,
ground_truth_hallucinated
```

All columns should have values (not empty).

---

## 🔄 Re-run Specific Steps

### Re-run just detection (keep answers):
```bash
# Edit scripts/04_run_pipeline.py to skip generation
# Or delete specific detector results and re-run
```

### Re-run just visualizations:
```bash
python scripts/05_create_visualizations.py
```

### Re-run single quality tier:
```python
# In scripts/04_run_pipeline.py, comment out unwanted tiers:
quality_tiers = ['high']  # Instead of ['high', 'medium', 'low']
```

---

## 📧 Getting Help

If stuck:
1. Check `outputs/` for logs
2. Run `test_detectors.py` to isolate issues
3. Check GPU with `nvidia-smi`
4. Verify config: `cat config/config.yaml`

Common issues:
- Empty detector columns → Detectors failed, check logs
- Out of memory → Reduce batch size or max_new_tokens
- Slow performance → Verify GPU is being used
- BM25 failing → Java not installed correctly

---

## ✨ Success Checklist

- [ ] GPU detected (`nvidia-smi` works)
- [ ] FAISS GPU working (`test_faiss_gpu.py` passes)
- [ ] BM25 index built (Linux advantage!)
- [ ] FAISS index built (~2 min with GPU)
- [ ] Pipeline completed all 3 tiers (~25 min)
- [ ] All detector columns filled (RAGAS, NLI, Lexical)
- [ ] Visualizations generated
- [ ] Results downloaded to local machine
- [ ] Instance destroyed (stop paying!)

**Total time: 30-45 minutes from zero to complete results** 🎉
