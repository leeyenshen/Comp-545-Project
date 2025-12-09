# Quick Start Guide: RAG Hallucination Detection

This guide will help you run the complete RAG hallucination detection pipeline.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, 16GB+ VRAM)
- 50GB+ free disk space

## Installation

### 1. Install Dependencies

```bash
cd "/Users/leeyenshen/Desktop/Comp 545 Project"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install additional NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### 2. Install Java (for Pyserini/BM25)

Pyserini requires Java 11+:

```bash
# On macOS with Homebrew
brew install openjdk@11

# On Ubuntu/Debian
sudo apt-get install openjdk-11-jdk

# Verify installation
java -version
```

## Running the Pipeline

The pipeline consists of 5 main steps:

### Step 1: Download Datasets (Week 1)

```bash
python scripts/01_download_datasets.py
```

**What it does:**
- Downloads NaturalQuestions dataset (1,000 samples)
- Downloads Wikipedia passages for retrieval corpus
- Saves data to `data/raw/`

**Expected output:**
- `data/raw/nq_questions.jsonl`
- `data/raw/wikipedia_passages.jsonl`

**Time:** ~15-30 minutes (depending on internet speed)

---

### Step 2: Build BM25 Index (Week 1)

```bash
python scripts/02_build_bm25_index.py
```

**What it does:**
- Converts Wikipedia passages to Pyserini format
- Builds BM25 (sparse) retrieval index
- Tests retrieval with sample query

**Expected output:**
- `data/indices/bm25_index/`
- Index files for BM25 search

**Time:** ~10-20 minutes

---

### Step 3: Build FAISS Index (Week 1)

```bash
python scripts/03_build_faiss_index.py
```

**What it does:**
- Creates dense embeddings using sentence-transformers
- Builds FAISS index for semantic search
- Tests retrieval with sample query

**Expected output:**
- `data/embeddings/passage_embeddings.npy`
- `data/embeddings/passage_metadata.pkl`
- `data/indices/faiss_index.bin`

**Time:** ~20-40 minutes (GPU recommended)

---

### Step 4: Run Full Pipeline (Weeks 2-3)

This is the main experiment script that:
1. Retrieves contexts at different quality tiers
2. Generates answers using LLM
3. Runs hallucination detection
4. Evaluates performance

```bash
python scripts/04_run_pipeline.py
```

**What it does:**
- For each quality tier (high, medium, low):
  - Retrieves documents with controlled distractor injection
  - Generates answers using Mistral-7B
  - Runs RAGAS, NLI, and Lexical detectors
  - Labels ground truth hallucinations
  - Saves results

**Expected output:**
- `outputs/results/results_high.jsonl`
- `outputs/results/results_medium.jsonl`
- `outputs/results/results_low.jsonl`
- `outputs/results/results_*.csv` (for easy viewing)

**Time:** ~2-4 hours (depends on GPU and number of samples)

**⚠️ Important Notes:**
- This step requires a GPU with sufficient VRAM (16GB+ recommended)
- Adjust `num_questions` in the script to process fewer samples for testing
- The script loads Mistral-7B with 8-bit quantization by default

---

### Step 5: Create Visualizations (Week 3)

```bash
python scripts/05_create_visualizations.py
```

**What it does:**
- Loads evaluation results
- Creates publication-quality plots:
  - Performance vs quality plots (for each metric)
  - All metrics comparison
  - Confusion matrices
  - AUROC heatmap
  - Retrieval quality distribution
- Generates LaTeX tables for paper

**Expected output:**
- `outputs/visualizations/*.png` (all plots)
- `outputs/results/results_table.csv`
- `outputs/results/results_table.tex` (for paper)

**Time:** ~2-5 minutes

---

## Evaluation Only (If Pipeline Already Run)

If you've already run the pipeline and just want to re-evaluate or re-visualize:

```bash
# Re-run evaluation
python src/evaluation/evaluator.py

# Re-create visualizations
python scripts/05_create_visualizations.py
```

## Testing Individual Components

### Test Retrieval

```bash
python src/retrieval/retriever.py
```

### Test Answer Generation

```bash
python src/generation/answer_generator.py
```

### Test RAGAS Detector

```bash
python src/detection/ragas_detector.py
```

### Test NLI Detector

```bash
python src/detection/nli_detector.py
```

### Test Lexical Detector

```bash
python src/detection/lexical_detector.py
```

## Configuration

Edit `config/config.yaml` to customize:

- Dataset selection and size
- Retrieval parameters (BM25, FAISS)
- Quality tier ratios
- Generation model and parameters
- Detection thresholds
- Output paths

## Project Structure

```
Comp 545 Project/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── raw/                     # Downloaded datasets
│   ├── processed/               # Processed data
│   ├── indices/                 # BM25 and FAISS indices
│   └── embeddings/              # Dense embeddings
├── src/
│   ├── retrieval/
│   │   └── retriever.py         # RAG retriever with distractor injection
│   ├── generation/
│   │   └── answer_generator.py # LLM answer generation
│   ├── detection/
│   │   ├── ragas_detector.py   # RAGAS detector
│   │   ├── nli_detector.py     # NLI-based detector
│   │   └── lexical_detector.py # Lexical overlap detector
│   └── evaluation/
│       └── evaluator.py         # Evaluation metrics
├── scripts/
│   ├── 01_download_datasets.py
│   ├── 02_build_bm25_index.py
│   ├── 03_build_faiss_index.py
│   ├── 04_run_pipeline.py       # Main pipeline
│   └── 05_create_visualizations.py
├── outputs/
│   ├── results/                 # Experiment results
│   └── visualizations/          # Plots and figures
├── paper/
│   ├── main.tex                 # LaTeX paper
│   └── references.bib           # Bibliography
├── requirements.txt
└── README.md
```

## Troubleshooting

### Out of Memory (GPU)

If you encounter GPU memory errors:

1. Reduce batch size in answer generation
2. Process fewer questions at a time
3. Use CPU instead (slower):
   ```python
   # In config.yaml, change:
   generation:
     device: "cpu"
   ```

### Pyserini Indexing Fails

If BM25 indexing fails:

1. Verify Java is installed: `java -version`
2. Check JAVA_HOME is set
3. Try running with explicit Java path

### RAGAS Errors

RAGAS requires an API key for some features. If you get API errors:

1. Set OpenAI API key (optional):
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```
2. Or modify RAGAS detector to use local models only

### Slow Performance

To speed up:

1. Reduce `num_questions` in pipeline script
2. Use GPU for all components
3. Reduce embedding dimensions in config
4. Use smaller LLM (e.g., Mistral-7B instead of 13B)

## Expected Runtime

For **50 questions** on a single GPU (RTX 3090/A100):

- Data download: ~20 min
- BM25 indexing: ~10 min
- FAISS indexing: ~30 min
- Pipeline (all 3 tiers): ~1-2 hours
- Visualization: ~2 min

**Total: ~2-3 hours**

For **1,000 questions**: multiply by ~20x

## Next Steps

After running the pipeline:

1. Review results in `outputs/results/`
2. Examine visualizations in `outputs/visualizations/`
3. Read the generated LaTeX table in `outputs/results/results_table.tex`
4. Compile the paper in `paper/main.tex`
5. Analyze failure modes and patterns

## Citation

If you use this code, please cite:

```bibtex
@article{yourname2024rag,
  title={Robustness of Hallucination Detection in Retrieval-Augmented Generation},
  author={Your Name},
  year={2024}
}
```

## License

MIT License (or your preferred license)

## Contact

For questions or issues:
- Email: your.email@university.edu
- GitHub: [repository URL]
