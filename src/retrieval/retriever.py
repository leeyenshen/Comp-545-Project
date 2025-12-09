"""
Retrieval module with distractor injection for controlled quality degradation
"""

import json
import random
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pyserini.search.lucene import LuceneSearcher
from typing import List, Dict, Tuple

class RAGRetriever:
    """
    Unified retriever supporting both BM25 and dense retrieval
    with controlled distractor injection
    """

    def __init__(self, config):
        self.config = config
        self.bm25_searcher = None
        self.faiss_index = None
        self.dense_model = None
        self.passages = None

    def load_bm25(self):
        """Load BM25 index"""
        index_path = Path(self.config['paths']['indices_dir']) / "bm25_index"
        if index_path.exists():
            self.bm25_searcher = LuceneSearcher(str(index_path))
            print(f"Loaded BM25 index from {index_path}")
        else:
            print(f"Warning: BM25 index not found at {index_path}")

    def load_faiss(self):
        """Load FAISS index and embeddings"""
        index_path = Path(self.config['paths']['indices_dir']) / "faiss_index.bin"
        metadata_path = Path(self.config['paths']['embeddings_dir']) / "passage_metadata.pkl"

        if index_path.exists() and metadata_path.exists():
            self.faiss_index = faiss.read_index(str(index_path))

            with open(metadata_path, 'rb') as f:
                self.passages = pickle.load(f)

            # Load dense retrieval model
            model_name = self.config['retrieval']['dense']['model_name']
            self.dense_model = SentenceTransformer(model_name)

            print(f"Loaded FAISS index from {index_path}")
        else:
            print(f"Warning: FAISS index or metadata not found")

    def retrieve_bm25(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve using BM25
        """
        if self.bm25_searcher is None:
            self.load_bm25()

        hits = self.bm25_searcher.search(query, k=k)

        results = []
        for hit in hits:
            results.append({
                "id": hit.docid,
                "score": hit.score,
                "text": json.loads(hit.raw)['contents']
            })

        return results

    def retrieve_dense(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve using dense embeddings (FAISS)
        """
        if self.faiss_index is None or self.dense_model is None:
            self.load_faiss()

        # Encode query
        query_embedding = self.dense_model.encode([query], convert_to_numpy=True)

        # Search
        distances, indices = self.faiss_index.search(query_embedding, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            passage = self.passages[idx]
            results.append({
                "id": passage['id'],
                "score": float(1.0 / (1.0 + dist)),  # Convert distance to similarity
                "text": f"{passage['title']}\n{passage['text']}",
                "title": passage['title']
            })

        return results

    def inject_distractors(
        self,
        relevant_docs: List[Dict],
        quality_tier: str
    ) -> List[Dict]:
        """
        Inject distractors based on quality tier

        Args:
            relevant_docs: List of relevant documents
            quality_tier: 'high', 'medium', or 'low'

        Returns:
            Mixed list of relevant and distractor documents
        """
        if self.passages is None:
            self.load_faiss()

        # Get ratios from config
        tier_config = self.config['retrieval']['quality_tiers'][quality_tier]
        relevant_ratio = tier_config['relevant_ratio']
        distractor_ratio = tier_config['distractor_ratio']

        # Calculate number of documents needed
        num_relevant = int(len(relevant_docs) * relevant_ratio)
        num_total = len(relevant_docs)
        num_distractors = num_total - num_relevant

        # Sample relevant documents
        if num_relevant < len(relevant_docs):
            relevant_sampled = random.sample(relevant_docs, num_relevant)
        else:
            relevant_sampled = relevant_docs

        # Sample distractors from passage pool
        distractors = []
        if num_distractors > 0:
            # Get random passages as distractors
            distractor_passages = random.sample(self.passages, min(num_distractors, len(self.passages)))

            for passage in distractor_passages:
                distractors.append({
                    "id": passage['id'],
                    "score": 0.0,  # Low score for distractors
                    "text": f"{passage['title']}\n{passage['text']}",
                    "is_distractor": True
                })

        # Combine and shuffle
        mixed_docs = relevant_sampled + distractors
        random.shuffle(mixed_docs)

        return mixed_docs

    def retrieve_with_quality_control(
        self,
        query: str,
        quality_tier: str = 'high',
        method: str = 'dense',
        k: int = 5
    ) -> Tuple[List[Dict], Dict]:
        """
        Retrieve documents with controlled quality degradation

        Args:
            query: Query string
            quality_tier: 'high', 'medium', or 'low'
            method: 'bm25' or 'dense'
            k: Number of documents to retrieve

        Returns:
            (documents, metadata)
        """
        # Retrieve relevant documents
        if method == 'bm25':
            relevant_docs = self.retrieve_bm25(query, k=k)
        else:
            relevant_docs = self.retrieve_dense(query, k=k)

        # Inject distractors based on quality tier
        mixed_docs = self.inject_distractors(relevant_docs, quality_tier)

        # Prepare metadata
        metadata = {
            "query": query,
            "quality_tier": quality_tier,
            "method": method,
            "num_relevant": len([d for d in mixed_docs if not d.get('is_distractor', False)]),
            "num_distractors": len([d for d in mixed_docs if d.get('is_distractor', False)]),
            "total_docs": len(mixed_docs)
        }

        return mixed_docs, metadata

def test_retriever(config):
    """Test the retriever with sample queries"""
    print("Testing RAG Retriever...")

    retriever = RAGRetriever(config)

    test_query = "What is artificial intelligence?"

    # Test all quality tiers
    for tier in ['high', 'medium', 'low']:
        print(f"\n{'='*60}")
        print(f"Quality Tier: {tier.upper()}")
        print('='*60)

        docs, metadata = retriever.retrieve_with_quality_control(
            test_query,
            quality_tier=tier,
            method='dense'
        )

        print(f"Query: {test_query}")
        print(f"Retrieved: {metadata['total_docs']} docs "
              f"({metadata['num_relevant']} relevant, {metadata['num_distractors']} distractors)")

        for i, doc in enumerate(docs[:3]):
            print(f"\n{i+1}. {'[DISTRACTOR]' if doc.get('is_distractor') else '[RELEVANT]'}")
            print(f"   Score: {doc['score']:.4f}")
            print(f"   Text: {doc['text'][:150]}...")

if __name__ == "__main__":
    import yaml

    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    test_retriever(config)
