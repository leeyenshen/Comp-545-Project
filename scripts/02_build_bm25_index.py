"""
Week 1: Build BM25 Index using Pyserini
Creates a BM25 index over Wikipedia passages for sparse retrieval
"""

import os
import json
import yaml
from pathlib import Path
import subprocess
from tqdm import tqdm

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_jsonl_for_indexing(config):
    """
    Convert Wikipedia passages to Pyserini-compatible JSONL format
    Each line should have: {"id": "...", "contents": "..."}
    """
    print("Preparing JSONL for Pyserini indexing...")

    input_path = Path(config['paths']['raw_data']) / "wikipedia_passages.jsonl"
    output_path = Path(config['paths']['processed_data']) / "wiki_for_bm25.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in tqdm(fin, desc="Converting format"):
            passage = json.loads(line)

            # Pyserini format
            pyserini_doc = {
                "id": passage['id'],
                "contents": f"{passage['title']}\n{passage['text']}"
            }

            fout.write(json.dumps(pyserini_doc) + '\n')

    print(f"Saved Pyserini-compatible JSONL to {output_path}")
    return output_path

def build_bm25_index(config, jsonl_path):
    """
    Build BM25 index using Pyserini
    """
    print("Building BM25 index with Pyserini...")

    # Create index directory
    index_dir = Path(config['paths']['indices_dir']) / "bm25_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Prepare command
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(jsonl_path.parent),
        "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "4",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw"
    ]

    print(f"Running command: {' '.join(cmd)}")

    try:
        # Run indexing
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"\nBM25 index built successfully at {index_dir}")
        return index_dir

    except subprocess.CalledProcessError as e:
        print(f"Error building BM25 index: {e}")
        print(f"stderr: {e.stderr}")
        return None

def test_bm25_retrieval(config, index_dir):
    """
    Test BM25 retrieval with a sample query
    """
    print("\nTesting BM25 retrieval...")

    try:
        from pyserini.search.lucene import LuceneSearcher

        # Initialize searcher
        searcher = LuceneSearcher(str(index_dir))

        # Test query
        test_query = "What is machine learning?"
        hits = searcher.search(test_query, k=5)

        print(f"\nTest query: '{test_query}'")
        print(f"Retrieved {len(hits)} documents:")

        for i, hit in enumerate(hits):
            print(f"\n{i+1}. Score: {hit.score:.4f}")
            print(f"   ID: {hit.docid}")
            print(f"   Content: {hit.raw[:200]}...")

        return True

    except Exception as e:
        print(f"Error testing retrieval: {e}")
        return False

def main():
    """Main function to orchestrate BM25 index building"""
    config = load_config()

    print("="*60)
    print("Week 1: Building BM25 Index with Pyserini")
    print("="*60)

    # Step 1: Prepare JSONL
    jsonl_path = prepare_jsonl_for_indexing(config)

    # Step 2: Build index
    index_dir = build_bm25_index(config, jsonl_path)

    if index_dir:
        # Step 3: Test retrieval
        test_bm25_retrieval(config, index_dir)

        print("\n" + "="*60)
        print("BM25 index building complete!")
        print(f"Index location: {index_dir}")
        print("="*60)
    else:
        print("\nFailed to build BM25 index")

if __name__ == "__main__":
    main()
