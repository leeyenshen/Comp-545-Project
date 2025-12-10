# RAGAS Options - Making It Work

## The Problem

RAGAS is timing out because:
1. It runs TinyLlama locally (slow on CPU/MPS)
2. It makes multiple LLM calls per sample (~3-5 calls)
3. 50 samples × 3 metrics × 5 calls = **750 LLM inferences**
4. Each taking 10-20 seconds = **2-3 hours just for RAGAS**

## Solutions (Pick One)

### Option 1: Use OpenAI API (FASTEST ✅)

**Setup:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Modify RAGAS detector to skip local models:**
In `src/detection/ragas_detector.py`, comment out local model initialization:
```python
def __init__(self, config):
    # ... existing code ...
    # self._init_local_models()  # Comment this out
    self.llm = None  # Will use OpenAI
    self.embeddings = None
```

**Cost:** ~$0.10-0.50 for the whole experiment (very cheap with GPT-3.5)

**Time:** ~5-10 minutes total

---

### Option 2: Reduce Sample Size (QUICK TEST)

Edit `config/config.yaml`:
```yaml
dataset:
  subset_size: 100  # Instead of 1000
```

Then only process 10 samples per tier:
```python
# In scripts/rerun_detectors_only.py, line ~155
formatted_qa = formatted_qa[:10]  # Only first 10
```

**Time:** ~20-30 minutes for 10 samples per tier

---

### Option 3: Skip Some RAGAS Metrics

Edit `config/config.yaml`:
```yaml
detection:
  ragas:
    metrics:
      - "faithfulness"  # Keep only this one
      # - "answer_relevancy"  # Comment out
      # - "context_precision"  # Comment out
```

**Time:** ~40-60 minutes (3x faster)

---

### Option 4: Run on Vast.ai GPU ⚡

RAGAS with GPU will be **much** faster:
- Local TinyLlama on CPU: ~15-20 sec/call
- TinyLlama on GPU: ~1-2 sec/call
- **10x faster**

**Time:** ~15-20 minutes on GPU

---

### Option 5: Use Simpler Faithfulness Metric

Create a simplified RAGAS that only checks faithfulness without LLM:

```python
# Simple faithfulness: check if answer tokens appear in context
def simple_faithfulness(context, answer):
    context_words = set(context.lower().split())
    answer_words = set(answer.lower().split())
    overlap = len(context_words & answer_words) / len(answer_words)
    return overlap
```

This is basically what Lexical detector does, so you could:
- **Use NLI + Lexical only** (both work great!)
- Skip RAGAS entirely
- Still have 2 working detectors for your paper

**Time:** ~5-10 minutes

---

## My Recommendation

### For Quick Results:
**Use Option 5** - Skip RAGAS, rely on NLI + Lexical
- Both work perfectly on your system
- NLI gives you entailment scores (0.992 accuracy!)
- Lexical gives you overlap scores
- Done in 10 minutes

### For Complete Results with RAGAS:
**Use Option 1** - OpenAI API
- Fast (5-10 min)
- Cheap ($0.10-0.50)
- Accurate
- Standard RAGAS implementation

### For Free RAGAS:
**Use Option 4** - Vast.ai GPU
- ~$0.15 total cost
- 15-20 minutes
- Full RAGAS with local models

---

## Current Fixes Applied

I've already optimized your RAGAS:
1. ✅ Increased timeout to 300s per item
2. ✅ Reduced max_workers to 2 (less parallelism = more stable)
3. ✅ Optimized TinyLlama settings (greedy decoding, fewer tokens)
4. ✅ Added `raise_exceptions=False` to continue on errors

This should help, but RAGAS will still be slow (~45-60 min) on macOS.

---

## Quick Decision Tree

```
Do you have OpenAI API key?
├─ YES → Use Option 1 (5-10 min) ✅
└─ NO → Do you need RAGAS for paper?
    ├─ YES → Use Vast.ai GPU (Option 4) - $0.15, 20 min
    └─ NO → Skip RAGAS (Option 5) - Use NLI + Lexical only
```

## Try Now

**With current fixes:**
```bash
./run_detectors_only.sh
```

**Expected:** ~45-60 minutes (improved from 2-3 hours, but still slow)

**Faster alternative:**
```bash
# Skip RAGAS, run only NLI + Lexical
# (I can create this script if you want)
```

What would you like to do?
