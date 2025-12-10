"""
Week 1: Build FAISS Index for Dense Retrieval
Creates dense embeddings and FAISS index for semantic retrieval
"""

import os
import json
import yaml
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_passages(config):
    """Load Wikipedia passages"""
    passages_path = Path(config['paths']['raw_data']) / "wikipedia_passages.jsonl"

    passages = []
    with open(passages_path, 'r') as f:
        for line in tqdm(f, desc="Loading passages"):
            passages.append(json.loads(line))

    return passages

def create_embeddings(config, passages):
    """
    Create dense embeddings for all passages using sentence-transformers
    """
    print("Creating dense embeddings...")

    # Load model
    model_name = config['retrieval']['dense']['model_name']
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Prepare texts
    texts = [f"{p['title']}\n{p['text']}" for p in passages]

    # Create embeddings
    print("Encoding passages...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )

    print(f"Created embeddings shape: {embeddings.shape}")

    # Save embeddings
    embeddings_dir = Path(config['paths']['embeddings_dir'])
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = embeddings_dir / "passage_embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")

    # Save passage metadata
    metadata_path = embeddings_dir / "passage_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(passages, f)
    print(f"Saved metadata to {metadata_path}")

    return embeddings

def build_faiss_index(config, embeddings):
    """
    Build FAISS index for fast similarity search
    Automatically uses GPU if available
    """
    print("Building FAISS index...")

    # Get embedding dimension
    dimension = embeddings.shape[1]

    # Create FAISS index (L2 distance)
    index_cpu = faiss.IndexFlatL2(dimension)

    # Add embeddings
    index_cpu.add(embeddings)

    print(f"FAISS index built with {index_cpu.ntotal} vectors")

    # Try to move to GPU if available
    num_gpus = faiss.get_num_gpus()
    if num_gpus > 0:
        print(f"🚀 {num_gpus} GPU(s) detected! Converting FAISS index to GPU...")
        try:
            res = faiss.StandardGpuResources()
            index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            print("✅ FAISS index running on GPU for faster search")

            # Save CPU version for portability
            index_dir = Path(config['paths']['indices_dir'])
            index_dir.mkdir(parents=True, exist_ok=True)
            index_path = index_dir / "faiss_index.bin"
            faiss.write_index(index_cpu, str(index_path))
            print(f"Saved CPU index to {index_path} (for portability)")

            return index_gpu
        except Exception as e:
            print(f"⚠️  GPU conversion failed: {e}")
            print("Falling back to CPU index")
            index = index_cpu
    else:
        print("⚠️  No GPU available, using CPU index")
        index = index_cpu

    # Save index
    index_dir = Path(config['paths']['indices_dir'])
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")

    return index

def test_faiss_retrieval(config, index, passages):
    """
    Test FAISS retrieval with a sample query
    """
    print("\nTesting FAISS retrieval...")

    # Load model
    model_name = config['retrieval']['dense']['model_name']
    model = SentenceTransformer(model_name)

    # Test query
    test_query = "What is machine learning?"
    query_embedding = model.encode([test_query], convert_to_numpy=True)

    # Search
    k = 5
    distances, indices = index.search(query_embedding, k)

    print(f"\nTest query: '{test_query}'")
    print(f"Retrieved {k} documents:")

    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        passage = passages[idx]
        print(f"\n{i+1}. Distance: {dist:.4f}")
        print(f"   Title: {passage['title']}")
        print(f"   Text: {passage['text'][:200]}...")

    return True

def main():
    """Main function to orchestrate FAISS index building"""
    config = load_config()

    print("="*60)
    print("Week 1: Building FAISS Index for Dense Retrieval")
    print("="*60)

    # Step 1: Load passages
    passages = load_passages(config)
    print(f"Loaded {len(passages)} passages")

    # Step 2: Create embeddings
    embeddings = create_embeddings(config, passages)

    # Step 3: Build FAISS index
    index = build_faiss_index(config, embeddings)

    # Step 4: Test retrieval
    test_faiss_retrieval(config, index, passages)

    print("\n" + "="*60)
    print("FAISS index building complete!")
    print("="*60)

if __name__ == "__main__":
    main()
