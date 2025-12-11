"""
Run this in Google Colab to build embeddings with free GPU
Then download the files to your Mac

Instructions:
1. Go to https://colab.research.google.com/
2. Create new notebook
3. Copy this entire script into a code cell
4. Runtime -> Change runtime type -> GPU (T4)
5. Run the cell
6. Download the generated files when done
"""

# Install dependencies
!pip install sentence-transformers faiss-cpu datasets tqdm

import json
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict

print("="*60)
print("BUILDING EMBEDDINGS ON COLAB GPU")
print("="*60)

# Step 1: Download Wikipedia passages
print("\nStep 1: Downloading 100k Wikipedia passages...")
try:
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True
    )

    passages = []
    title_count = defaultdict(int)
    max_passage_length = 500

    print("Processing Wikipedia articles...")
    for i, article in enumerate(tqdm(dataset, total=200000, desc="Sampling")):
        if len(passages) >= 100000:
            break

        title = article.get('title', 'Unknown')
        text = article.get('text', '')

        # Split long articles
        if len(text) > max_passage_length:
            chunks = [text[j:j+max_passage_length] for j in range(0, len(text), max_passage_length)]
            for chunk in chunks[:5]:
                if len(passages) >= 100000:
                    break
                if chunk.strip():
                    passages.append({
                        'id': f"wiki_{len(passages)}",
                        'title': title,
                        'text': chunk.strip(),
                        'full_text': chunk.strip()
                    })
                    title_count[title] += 1
        else:
            if text.strip():
                passages.append({
                    'id': f"wiki_{len(passages)}",
                    'title': title,
                    'text': text.strip(),
                    'full_text': text.strip()
                })
                title_count[title] += 1

    print(f"\n✓ Collected {len(passages):,} passages")
    print(f"✓ Unique articles: {len(title_count):,}")

except Exception as e:
    print(f"❌ Download failed: {e}")
    print("Exiting...")
    raise

# Step 2: Save passages
print("\nStep 2: Saving passages...")
with open('wikipedia_passages.jsonl', 'w') as f:
    for passage in passages:
        f.write(json.dumps(passage) + '\n')
print(f"✓ Saved to wikipedia_passages.jsonl ({len(passages):,} passages)")

# Step 3: Create embeddings
print("\nStep 3: Creating embeddings with GPU...")
print("Loading model: multi-qa-mpnet-base-dot-v1")
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

texts = [f"{p['title']}\n{p['text']}" for p in passages]

print(f"Encoding {len(texts):,} passages with GPU...")
print("Estimated time: ~15-20 minutes on Colab GPU")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=128,  # Larger batch for GPU
    convert_to_numpy=True,
    normalize_embeddings=False
)

print(f"✓ Created embeddings shape: {embeddings.shape}")

# Step 4: Save embeddings
print("\nStep 4: Saving embeddings...")
np.save('passage_embeddings.npy', embeddings)
print("✓ Saved passage_embeddings.npy")

# Step 5: Save metadata
with open('passage_metadata.pkl', 'wb') as f:
    pickle.dump(passages, f)
print("✓ Saved passage_metadata.pkl")

# Step 6: Build FAISS index
print("\nStep 5: Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"✓ FAISS index built with {index.ntotal:,} vectors")

# Step 7: Save FAISS index
faiss.write_index(index, 'faiss_index.bin')
print("✓ Saved faiss_index.bin")

# Step 8: Show file sizes
from pathlib import Path
files = {
    'wikipedia_passages.jsonl': Path('wikipedia_passages.jsonl'),
    'passage_embeddings.npy': Path('passage_embeddings.npy'),
    'passage_metadata.pkl': Path('passage_metadata.pkl'),
    'faiss_index.bin': Path('faiss_index.bin')
}

print("\n" + "="*60)
print("BUILD COMPLETE!")
print("="*60)
print("\nGenerated files:")
for name, path in files.items():
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  {name}: {size_mb:.1f} MB")

print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print("""
1. Download these 4 files from Colab (click folder icon on left):
   - wikipedia_passages.jsonl
   - passage_embeddings.npy
   - passage_metadata.pkl
   - faiss_index.bin

2. On your Mac, move files to correct locations:
   mv wikipedia_passages.jsonl data/raw/
   mv passage_embeddings.npy data/embeddings/
   mv passage_metadata.pkl data/embeddings/
   mv faiss_index.bin data/indices/

3. Run the pipeline:
   ./run_pipeline_safe.sh
""")

print("\n✓ All done! Download the files and continue on your Mac.")
