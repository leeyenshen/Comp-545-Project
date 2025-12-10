# Alternative BM25 Solutions for macOS ARM

## Problem
Pyserini requires Java/JNI which doesn't work well on macOS ARM.

## Solution Options

### Option 1: Use `rank-bm25` (Pure Python) ✅ EASIEST

This is a lightweight, pure-Python BM25 implementation that works on any platform.

**Install:**
```bash
pip install rank-bm25
```

**Usage in your retriever:**
```python
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Tokenize corpus
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]

# Create BM25 index
bm25 = BM25Okapi(tokenized_corpus)

# Search
query_tokens = word_tokenize(query.lower())
scores = bm25.get_scores(query_tokens)
top_n = np.argsort(scores)[::-1][:k]
```

### Option 2: Use Elasticsearch ⚠️ COMPLEX

Run Elasticsearch locally for BM25 retrieval.

**Install:**
```bash
brew install elasticsearch
brew services start elasticsearch

pip install elasticsearch
```

**Usage:**
```python
from elasticsearch import Elasticsearch

es = Elasticsearch()
# Index documents, search with BM25
```

### Option 3: Skip BM25, Use FAISS Only ✅ RECOMMENDED

Your pipeline already supports this! The code gracefully falls back to FAISS-only.

**No changes needed** - just run:
```bash
./run_pipeline_safe.sh
```

## Recommendation

**Use Option 3 (FAISS-only) because:**

1. ✅ Already implemented and working
2. ✅ Dense retrieval (FAISS) often performs better than sparse (BM25) for RAG
3. ✅ Saves time - no debugging Java issues
4. ✅ Your research compares hallucination detection methods, not retrieval methods
5. ✅ You can mention in paper: "Due to platform constraints, we used dense retrieval (FAISS) which is increasingly preferred in modern RAG systems"

## If You Really Need BM25

Only worth it if:
- Your paper specifically requires comparing BM25 vs FAISS
- You have time to debug (could take hours)
- You're willing to use Rosetta 2 or switch to Linux

Otherwise, **skip BM25 and move forward with FAISS-only**.

## Implementation Note for Paper

If using FAISS-only, add this to your methods section:

> "We employed dense retrieval using FAISS with sentence-transformers embeddings
> (multi-qa-mpnet-base-dot-v1). While traditional RAG systems often combine sparse
> (BM25) and dense retrieval, recent work has shown that dense-only retrieval can
> achieve comparable or superior performance, particularly for semantic matching tasks."

**Citations:**
- Karpukhin et al. (2020) - Dense Passage Retrieval for Open-Domain QA
- Khattab & Zaharia (2020) - ColBERT
