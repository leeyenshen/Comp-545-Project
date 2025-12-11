# Setup 100K Wikipedia Corpus

This guide helps you expand your Wikipedia corpus from 224 articles (10k passages) to a diverse 100k passages covering thousands of topics.

## Why This Is Needed

**Current Problem:**
- Your corpus has only 224 Wikipedia articles
- Natural Questions asks about diverse topics not in your corpus
- FAISS retrieves "least irrelevant" passages (still completely irrelevant)
- LLM correctly says "don't know" for 100% of questions
- AUROC = 0 (can't calculate with only one class)

**After This Setup:**
- 100k passages covering thousands of topics
- Much better question coverage
- Mix of answerable and unanswerable questions
- Mix of faithful and hallucinated answers
- Meaningful AUROC scores

## Time Estimates

| Step | Duration | Description |
|------|----------|-------------|
| **1. Download** | 10-15 min | Download 100k Wikipedia passages from wiki_dpr |
| **2. Embed** | 3-5 hours | Generate embeddings using MPS (Apple Silicon GPU) |
| **3. Index** | 5-10 min | Build FAISS index |
| **4. Pipeline** | 7-8 hours | Run full pipeline (3 tiers × 50 questions) |
| **TOTAL** | **11-14 hours** | Can run overnight |

## Quick Start

### Option A: Automated Setup (Recommended)

Run the automated setup script:

```bash
./setup_corpus_100k.sh
```

This will:
1. Download 100k diverse Wikipedia passages
2. Generate embeddings (uses Apple Silicon GPU automatically)
3. Build FAISS index
4. Show you next steps

Then run the pipeline:

```bash
./run_pipeline_safe.sh              # All 3 tiers (~7-8 hours)
# OR
./run_pipeline_single_tier.sh       # Just HIGH tier (~2.5 hours)
# OR
./test_single_question.sh           # Quick test (~5 min)
```

### Option B: Manual Step-by-Step

If you prefer to run steps manually:

**Step 1: Download corpus**
```bash
source venv/bin/activate
python scripts/download_wiki_dpr.py
```

**Step 2: Build FAISS index (includes embedding generation)**
```bash
python scripts/03_build_faiss_index.py
```

**Step 3: Run pipeline**
```bash
./run_pipeline_safe.sh
```

## What's Changed

The setup makes these changes:

1. **Downloads new Wikipedia corpus** (`data/raw/wikipedia_passages.jsonl`)
   - Old corpus backed up to `.backup`
   - New corpus: 100k passages instead of 10k
   - Diverse sampling for better coverage

2. **Generates new embeddings** (`data/embeddings/`)
   - Uses Apple Silicon GPU (MPS) for faster generation
   - 100k passages × 768 dimensions
   - Takes 3-5 hours on Mac

3. **Builds new FAISS index** (`data/indices/faiss_index.bin`)
   - Old index backed up automatically
   - New index: 100k vectors

4. **Pipeline now uses FAISS by default**
   - Changed `use_targeted=False` (was True)
   - Changed `use_filtered=False` (was True)
   - Uses real semantic search on 100k corpus

## Monitoring Progress

### During Download (10-15 min)
```
Loading dataset from HuggingFace...
Total passages available: 21,097,025
Sampling every 211th passage for diversity...
Sampling passages: 100%|████████| 100000/100000
```

### During Embedding (3-5 hours)
```
Loading model: multi-qa-mpnet-base-dot-v1
✓ Using Apple Silicon GPU (MPS) for embedding generation
Encoding 100,000 passages...
Batches: 100%|████████| 1563/1563 [3:45:23<00:00, 8.65s/it]
```

**Tip:** This is the longest step. You can run it overnight or while working on other things.

### During FAISS Index (5-10 min)
```
Building FAISS index...
FAISS index built with 100000 vectors
Saved FAISS index to data/indices/faiss_index.bin
```

### During Pipeline (7-8 hours)
```
RUNNING FULL PIPELINE - QUALITY TIER: HIGH
Retrieval Phase - Quality Tier: HIGH
Using FAISS retrieval (semantic search)
Loaded FAISS index from data/indices/faiss_index.bin
Retrieving (high): 100%|████████| 50/50 [00:05<00:00]

Answer Generation Phase
Generating answers: 100%|████████| 50/50 [2:15:46<00:00, 162.93s/it]
```

## Expected Results After Setup

With the new 100k corpus, you should see:

### Better Retrieval
- Retrieved contexts are **actually relevant** to questions
- Not just "least irrelevant" documents
- Proper quality tier degradation (high/medium/low)

### Mix of Answers
- Some questions: LLM answers correctly (faithful)
- Some questions: LLM hallucinates despite good context
- Some questions: LLM says "don't know" (no relevant passages)

### Meaningful Evaluation
- **~30-60% hallucinated** (instead of 100%)
- **~40-70% faithful** (instead of 0%)
- **AUROC > 0** (actual discrimination between classes)
- Detectors can be properly evaluated

## Troubleshooting

### "ModuleNotFoundError: No module named 'datasets'"

Install the datasets library:
```bash
source venv/bin/activate
pip install datasets
```

### "MPS not available" warning

The script will fall back to CPU (slower but still works). MPS requires:
- Mac with Apple Silicon (M1/M2/M3)
- macOS 12.3+
- PyTorch with MPS support

### Embedding generation is too slow

If it's taking much longer than 5 hours:
- Check Activity Monitor: `sentence-transformers` should use GPU
- Reduce corpus size: Edit `download_wiki_dpr.py` line with `target_passages=50000`
- Use CPU batching: Edit `03_build_faiss_index.py` line with `batch_size=16`

### Out of memory during embedding

Reduce batch size in `scripts/03_build_faiss_index.py`:
```python
batch_size = 32 if device == "mps" else 16  # Reduce these numbers
```

### Pipeline still gets 100% hallucinations

Check that:
1. New corpus was actually downloaded: `wc -l data/raw/wikipedia_passages.jsonl` shows 100000
2. New FAISS index exists: `ls -lh data/indices/faiss_index.bin` shows ~380MB
3. Pipeline loads the new index: Check for "Loaded FAISS index" in output
4. Test retrieval works: `./test_single_question.sh`

## Verify Setup Worked

After setup completes, verify with a single question test:

```bash
./test_single_question.sh
```

You should see:
- ✅ **Context retrieved:** 5 documents (not empty!)
- ✅ **Context relevant:** Related to the question
- ✅ **Answer quality:** LLM actually tries to answer (not just "don't know")

If you see:
- ❌ Empty contexts → FAISS index not loaded correctly
- ❌ Irrelevant contexts → Corpus still too small or download failed
- ❌ All "don't know" answers → Check contexts are being passed to LLM

## Next Steps After Setup

Once setup completes successfully:

1. **Test with 1 question** (~5 min)
   ```bash
   ./test_single_question.sh
   ```

2. **Run HIGH tier only** (~2.5 hours)
   ```bash
   ./run_pipeline_single_tier.sh
   ```

3. **Check results look good**
   - Open `outputs/results/results_high.csv`
   - Verify contexts are present and relevant
   - Verify mix of faithful/hallucinated answers

4. **Run full pipeline** (~7-8 hours)
   ```bash
   ./run_pipeline_safe.sh
   ```

5. **Analyze results**
   - AUROC heatmaps should show non-zero values
   - Detectors can be properly compared
   - Ready for research analysis!

## Questions?

If something doesn't work as expected, check:
- All steps completed without errors
- File sizes match expected (100k passages = ~300-400 MB JSONL)
- FAISS index size is reasonable (~380 MB for 100k passages)
- Test retrieval works before running full pipeline
