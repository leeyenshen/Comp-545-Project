# RAG Pipeline Issues Analysis and Proposed Fixes

## Executive Summary

The RAG hallucination detection pipeline has a fundamental **data mismatch issue** where the questions from Natural Questions dataset reference Wikipedia articles that don't exist in the retrieval corpus. This causes poor retrieval quality, which cascades into detection issues.

---

## Issue 1: Data Mismatch Between Questions and Knowledge Base

### Problem
- **Questions**: Downloaded from `nq_open` dataset (Natural Questions)
- **Knowledge Base**: Random 10,000 Wikipedia passages from `wiki_dpr` dataset
- **Mismatch**: Questions reference specific Wikipedia articles that are NOT in the 10k passage sample

### Example
- **Question**: "where did they film hot tub time machine"
- **Expected Answer**: "Fernie Alpine Resort"
- **Required Article**: Wikipedia article about "Hot Tub Time Machine" movie
- **Reality**: This article is NOT in the 10,000 passage sample
- **Result**: FAISS retrieves completely irrelevant documents (e.g., "BBC Television")

### Evidence
```bash
# Search for "Hot Tub Time Machine" in passages
$ grep -i "hot tub" data/raw/wikipedia_passages.jsonl
# Result: NO MATCHES

# Search for ground truth answer
$ grep -i "fernie" data/raw/wikipedia_passages.jsonl
# Result: NO MATCHES

# But BBC Television exists
$ grep -i "bbc television" data/raw/wikipedia_passages.jsonl
# Result: MULTIPLE MATCHES
```

### Root Cause
In [scripts/01_download_datasets.py](scripts/01_download_datasets.py:147-177):
```python
# Downloads NQ questions
download_natural_questions(config)  # Downloads questions expecting full Wikipedia

# Downloads random 10k passages
wiki = load_dataset("wiki_dpr", "psgs_w100.nq.exact", split="train")
wiki_sample = wiki.select(range(min(max_passages, wiki.num_rows)))  # Random 10k
```

The `wiki_dpr` dataset contains ~21M passages, but we only take the first 10,000. Natural Questions expects access to the full corpus or at least the relevant passages for the selected questions.

---

## Issue 2: Retrieval Quality Degradation

### Problem
When FAISS cannot find relevant documents (because they don't exist), it returns the "closest" documents by embedding similarity, which are often completely unrelated.

### Example from Test Run
```
Question: "where did they film hot tub time machine"
Retrieved Context: "BBC Television is a service of the British Broadcasting Corporation..."
```

### Why This Happens
1. FAISS uses cosine similarity on embeddings
2. When no relevant documents exist, it still returns k=5 documents
3. These are the "least irrelevant" documents, but still irrelevant
4. The embedding model tries to find semantic similarity, but fails without proper content

### Current Flow
```
Query: "hot tub time machine filming location"
  ↓ (embedding)
Query Embedding: [0.23, -0.45, 0.12, ...]
  ↓ (FAISS search in 10k passages)
No relevant passages found
  ↓ (returns closest matches by L2 distance)
Returns: ["BBC Television", "University of Notre Dame", ...]
  ↓
Model receives irrelevant context
  ↓
Model correctly says: "I don't know"
```

---

## Issue 3: Hallucination Detection Misclassification

### Problem
The model's response "I don't know" is being flagged as a hallucination by RAGAS and ground truth comparison, even though it's the correct response to insufficient context.

### Test Results
```
Answer: "don't know. The provided context does not mention where they filmed 'Hot Tub Time Machine.'"

Detection Results:
  RAGAS hallucinated: True      ← INCORRECT
  NLI hallucinated: False       ← CORRECT
  Lexical hallucinated: False   ← CORRECT
  Ground truth hallucinated: True ← INCORRECT
```

### Why RAGAS Flags "I Don't Know"
From [src/detection/ragas_detector.py](src/detection/ragas_detector.py:196-212):
```python
def classify_hallucination(self, faithfulness_score: float, threshold: float = 0.5) -> bool:
    # Lower faithfulness = more hallucination
    return faithfulness_score < threshold
```

RAGAS `faithfulness` metric evaluates if the answer is supported by the context. When the model says "I don't know", RAGAS interprets this as:
- The context doesn't explicitly state "the answer is unknown"
- Therefore, the answer is not faithful to the context
- Result: Low faithfulness score → flagged as hallucination

**This is technically correct from RAGAS's perspective**, but philosophically wrong. Abstaining from answering due to insufficient information should NOT be considered a hallucination.

### Why Ground Truth Comparison Flags It
From [scripts/04_run_pipeline.py](scripts/04_run_pipeline.py:188-210):
```python
def label_ground_truth(results):
    for result in results:
        answer_lower = result['answer'].lower()
        ground_truth_lower = str(result['ground_truth']).lower()

        # Check if ground truth appears in answer
        is_faithful = ground_truth_lower in answer_lower or answer_lower in ground_truth_lower

        result['ground_truth_hallucinated'] = not is_faithful
```

Since "I don't know" doesn't contain "Fernie Alpine Resort", it's marked as hallucinated. But this is actually a **retrieval failure**, not a generation hallucination.

---

## Proposed Fixes

### Fix 1: Align Questions with Available Knowledge (RECOMMENDED)

**Approach**: Only use questions whose answers can be found in the available 10k passages.

**Implementation**:
```python
# In scripts/01_download_datasets.py
def filter_answerable_questions(questions, passages):
    """
    Filter questions to only those answerable by available passages
    """
    # Build search index of passage titles
    passage_titles = {p['title'].lower() for p in passages}

    # For NQ, we can use the question to infer if answer might be present
    # Or better: use the document_title if available in NQ dataset
    answerable = []
    for q in questions:
        # Check if question's topic appears in passages
        # This is heuristic but better than nothing
        if might_be_answerable(q, passage_titles):
            answerable.append(q)

    return answerable
```

**Pros**:
- Quick fix
- Works with current 10k passage corpus
- Ensures valid evaluation

**Cons**:
- Reduces number of test questions
- Questions might still be partially unanswerable

### Fix 2: Download Full Wikipedia Corpus (IDEAL BUT EXPENSIVE)

**Approach**: Use the complete wiki_dpr dataset (~21M passages).

**Implementation**:
```python
# In scripts/01_download_datasets.py
def download_wikipedia_passages(config):
    wiki = load_dataset("wiki_dpr", "psgs_w100.nq.exact", split="train")
    # Don't sample - use all passages (or use much larger sample like 1M)
    wiki_passages = []
    for passage in tqdm(wiki):
        wiki_passages.append({
            "id": passage['id'],
            "title": passage['title'],
            "text": passage['text'],
            "full_text": passage['text']
        })
```

**Pros**:
- Proper evaluation
- Questions will have relevant context
- Realistic RAG scenario

**Cons**:
- Large storage requirement (~10-20GB)
- Slow FAISS indexing
- High memory usage

### Fix 3: Download Question-Specific Passages

**Approach**: For each question, download only the relevant Wikipedia passages.

**Implementation**:
```python
def download_relevant_passages_for_questions(questions):
    """
    For NQ dataset, extract the document titles/IDs and download only those passages
    """
    # NQ dataset includes document context in some versions
    # Or use Wikipedia API to fetch specific articles
    relevant_passages = []
    for q in questions:
        if 'document_title' in q:
            passage = fetch_wikipedia_article(q['document_title'])
            relevant_passages.append(passage)
    return relevant_passages
```

**Pros**:
- Small dataset size
- Fast indexing
- Guaranteed relevant context
- Good for testing

**Cons**:
- Doesn't test retrieval quality (passages are pre-selected)
- Not realistic RAG scenario
- Requires Wikipedia API access or special NQ dataset version

### Fix 4: Use Different Dataset

**Approach**: Use a QA dataset that comes with its own corpus (e.g., SQuAD, MS MARCO).

**Implementation**:
```python
# Use SQuAD v2 which includes contexts
dataset = load_dataset("squad_v2")
# Questions and contexts are paired
```

**Pros**:
- Self-contained
- No mismatch issues
- Includes "unanswerable" questions

**Cons**:
- Different from Natural Questions
- May not match research requirements

### Fix 5: Improve Hallucination Detection for "I Don't Know" Responses

**Approach**: Add special handling for abstention responses.

**Implementation**:
```python
# In src/detection/ragas_detector.py
def classify_hallucination(self, answer: str, faithfulness_score: float, threshold: float = 0.5) -> bool:
    # Check if answer is abstention
    abstention_phrases = [
        "i don't know",
        "i do not know",
        "cannot answer",
        "insufficient information",
        "not mentioned in the context"
    ]

    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in abstention_phrases):
        # Abstention is NOT a hallucination
        return False

    # Otherwise, use faithfulness score
    return faithfulness_score < threshold
```

```python
# In scripts/04_run_pipeline.py
def label_ground_truth(results):
    for result in results:
        answer_lower = result['answer'].lower()

        # Check for abstention
        if any(phrase in answer_lower for phrase in ["don't know", "do not know", "cannot answer"]):
            # Abstention should not be marked as hallucination
            # Instead, mark as "unanswerable_by_context"
            result['ground_truth_hallucinated'] = None  # or False
            result['is_abstention'] = True
            continue

        # ... rest of original logic
```

**Pros**:
- Handles abstention correctly
- More nuanced evaluation
- Recognizes retrieval failures vs generation hallucinations

**Cons**:
- Doesn't fix root cause (data mismatch)
- Heuristic-based (may miss some abstentions)

---

## Recommended Implementation Plan

### Phase 1: Quick Fix (1-2 hours)
1. Implement **Fix 5** - Handle "I don't know" responses correctly
2. Add logging to track retrieval quality metrics
3. Document the data mismatch issue

### Phase 2: Proper Fix (4-8 hours)
Choose one of:
- **Fix 1** - Filter questions to answerable ones (if dataset is flexible)
- **Fix 2** - Download larger passage corpus (if resources allow)
- **Fix 3** - Download question-specific passages (if testing focused)

### Phase 3: Evaluation Improvements (2-4 hours)
1. Add retrieval quality metrics:
   - Retrieval recall (are relevant docs in top-k?)
   - Average similarity score
   - Percentage of "no relevant context" cases
2. Separate evaluation into:
   - Retrieval quality
   - Generation quality (given good context)
   - Hallucination detection accuracy
3. Create visualizations showing the cascade of errors

---

## Additional Recommendations

### 1. Add Retrieval Confidence Threshold
Don't generate answers when retrieval confidence is too low:
```python
def retrieve_with_confidence(self, query, k=5, min_confidence=0.5):
    docs, scores = self.retrieve(query, k)

    if max(scores) < min_confidence:
        return None, "No relevant documents found"

    return docs, scores
```

### 2. Create Separate Test Sets
- **Test Set A**: Questions with relevant passages (tests generation + detection)
- **Test Set B**: Questions without relevant passages (tests abstention behavior)
- **Test Set C**: Questions with mixed relevant/irrelevant context (tests quality tiers)

### 3. Improve Context Quality Metrics
Track not just number of distractors, but also:
- Semantic similarity of distractors to query
- Presence of answer in retrieved context
- Position of relevant docs in ranking

---

## Summary

| Issue | Severity | Root Cause | Recommended Fix |
|-------|----------|------------|-----------------|
| Data Mismatch | **Critical** | Random 10k passages don't match NQ questions | Fix 1, 2, or 3 |
| Poor Retrieval | **High** | Consequence of data mismatch | Depends on data fix |
| "I Don't Know" Misclassified | **Medium** | Detection logic doesn't handle abstention | Fix 5 |
| Ground Truth Comparison Issues | **Medium** | Simple string matching inadequate | Fix 5 + better evaluation |

**Priority**: Fix the data mismatch first (Fix 1, 2, or 3), then implement Fix 5 for better detection logic.
