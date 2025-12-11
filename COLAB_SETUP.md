# Build Embeddings on Google Colab (Free GPU)

**Time: ~20-30 minutes total** (vs 4-6 hours on your Mac CPU)

This guide shows you how to build the embeddings on Google Colab's free GPU and download the files to your Mac.

## Why Use Colab?

- ✅ **Free GPU** (NVIDIA T4)
- ✅ **Fast** (~15-20 min vs 4-6 hours on Mac CPU)
- ✅ **No local resources** used
- ✅ **Easy to use**

## Step-by-Step Instructions

### 1. Open Google Colab

Go to: https://colab.research.google.com/

Click **"New Notebook"**

### 2. Enable GPU

- Click **Runtime** → **Change runtime type**
- Set **Hardware accelerator** to **GPU** (should show T4)
- Click **Save**

### 3. Copy and Run Script

- Open `colab_build_embeddings.py` from your project folder
- Copy the **entire file contents**
- Paste into the Colab code cell
- Click the **Play button** ▶️ or press `Shift+Enter`

### 4. Wait for Completion (~20-30 minutes)

The script will:
```
Step 1: Downloading 100k Wikipedia passages... (~5 min)
Step 2: Saving passages... (~1 min)
Step 3: Creating embeddings with GPU... (~15-20 min)
Step 4: Saving embeddings... (~1 min)
Step 5: Building FAISS index... (~1 min)
```

Watch for this final message:
```
BUILD COMPLETE!
✓ All done! Download the files and continue on your Mac.
```

### 5. Download Generated Files

Click the **folder icon** 📁 on the left sidebar

You'll see 4 files:
- `wikipedia_passages.jsonl` (~300 MB)
- `passage_embeddings.npy` (~300 MB)
- `passage_metadata.pkl` (~15 MB)
- `faiss_index.bin` (~380 MB)

**Right-click each file** → **Download**

All files will download to your Mac's Downloads folder.

### 6. Move Files to Your Project

Open Terminal and run:

```bash
cd "/Users/leeyenshen/Desktop/Comp 545 Project"

# Create directories if needed
mkdir -p data/raw data/embeddings data/indices

# Move files from Downloads (adjust path if needed)
mv ~/Downloads/wikipedia_passages.jsonl data/raw/
mv ~/Downloads/passage_embeddings.npy data/embeddings/
mv ~/Downloads/passage_metadata.pkl data/embeddings/
mv ~/Downloads/faiss_index.bin data/indices/
```

### 7. Verify Files Are In Place

```bash
ls -lh data/raw/wikipedia_passages.jsonl
ls -lh data/embeddings/passage_embeddings.npy
ls -lh data/embeddings/passage_metadata.pkl
ls -lh data/indices/faiss_index.bin
```

You should see:
```
data/raw/wikipedia_passages.jsonl      ~300 MB
data/embeddings/passage_embeddings.npy ~300 MB
data/embeddings/passage_metadata.pkl   ~15 MB
data/indices/faiss_index.bin          ~380 MB
```

### 8. Run the Pipeline!

Now you can run the pipeline with your new 100k corpus:

```bash
# Quick test with 1 question
./test_single_question.sh

# If that works, run the full pipeline
./run_pipeline_safe.sh
```

## Troubleshooting

### Colab disconnects during execution

If Colab disconnects:
- **Don't worry!** Your progress is saved
- Reconnect and check if files exist (folder icon)
- If files are there, just download them
- If not, you may need to rerun (keep browser tab active)

### "Resource exhausted" error in Colab

Try these fixes:
1. **Restart runtime**: Runtime → Restart runtime
2. **Reduce batch size**: Change line with `batch_size=128` to `batch_size=64`
3. **Use smaller corpus**: Change `100000` to `50000` on line where it says `if len(passages) >= 100000`

### Download is slow

The files are ~1 GB total. On a good connection:
- Should take 5-15 minutes to download all 4 files
- Download them one at a time if batch download fails

### Files already exist in my project

That's fine! The new files will replace the old ones:
```bash
# Backup old files first if you want
mv data/raw/wikipedia_passages.jsonl data/raw/wikipedia_passages.jsonl.old
mv data/embeddings/passage_embeddings.npy data/embeddings/passage_embeddings.npy.old
mv data/embeddings/passage_metadata.pkl data/embeddings/passage_metadata.pkl.old
mv data/indices/faiss_index.bin data/indices/faiss_index.bin.old

# Then move new files
mv ~/Downloads/wikipedia_passages.jsonl data/raw/
# ... etc
```

## Alternative: Use Vast.ai (If Colab Doesn't Work)

If Colab gives you issues, you can use Vast.ai:

1. Sign up at https://vast.ai/
2. Add $10 credit
3. Rent a machine with GPU (~$0.10-0.30/hour)
4. Upload the script
5. Run and download files
6. Total cost: ~$0.10 for 30 minutes

See `VASTAI_SETUP.md` for detailed instructions.

## Expected Results

After downloading and running the pipeline with the new corpus:

### Before (old 10k corpus, 224 articles):
- ❌ Contexts: Mostly irrelevant (limited coverage)
- ❌ Answers: 100% "don't know"
- ❌ AUROC: 0 (can't calculate)

### After (new 100k corpus, thousands of articles):
- ✅ Contexts: Relevant to questions
- ✅ Answers: Mix of faithful (~40-70%) and hallucinated (~30-60%)
- ✅ AUROC: > 0 (proper discrimination)
- ✅ Meaningful detector comparison

## Questions?

If something doesn't work:
1. Check the Colab output for error messages
2. Make sure GPU is enabled (Runtime → Change runtime type)
3. Try restarting the runtime and running again
4. Check that all 4 files downloaded successfully
5. Verify file sizes match expected sizes (~1 GB total)

Good luck! This should be much faster than local CPU embedding.
