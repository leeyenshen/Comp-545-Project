"""
Targeted retrieval using pre-computed question-passage mappings.

This bypasses FAISS and directly retrieves passages we know are relevant,
then applies quality tier distractor injection.

This ensures the "high quality" tier actually has relevant documents.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


class TargetedRetriever:
    """
    Retrieves passages using pre-computed question-passage mappings
    instead of semantic search.
    """

    def __init__(self, config):
        self.config = config
        self.passages = None
        self.passage_dict = None
        self.question_mapping = None

    def load_data(self):
        """Load passages and question-passage mapping"""
        # Load passages
        passages_path = Path(self.config['paths']['raw_data']) / "wikipedia_passages.jsonl"
        self.passages = []
        with open(passages_path, 'r') as f:
            for line in f:
                passage = json.loads(line)
                self.passages.append(passage)

        # Create lookup dict
        self.passage_dict = {p['id']: p for p in self.passages}

        # Load question-passage mapping
        mapping_path = Path(self.config['paths']['raw_data']) / "question_passage_mapping.json"
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                self.question_mapping = json.load(f)
        else:
            print(f"Warning: Question mapping not found at {mapping_path}")
            self.question_mapping = {}

        print(f"Loaded {len(self.passages)} passages and {len(self.question_mapping)} question mappings")

    def get_relevant_passages(self, question: str) -> List[Dict]:
        """
        Get relevant passages for a question using the pre-computed mapping

        Args:
            question: Question text

        Returns:
            List of relevant passage dicts
        """
        if self.passages is None:
            self.load_data()

        # Look up relevant passage IDs
        relevant_ids = self.question_mapping.get(question, [])

        if not relevant_ids:
            print(f"Warning: No relevant passages found for question: {question}")
            return []

        # Retrieve passages
        relevant_passages = []
        for pid in relevant_ids:
            if pid in self.passage_dict:
                passage = self.passage_dict[pid]
                relevant_passages.append({
                    "id": passage['id'],
                    "score": 1.0,  # Perfect match
                    "text": f"{passage['title']}\n{passage['text']}",
                    "title": passage['title'],
                    "is_distractor": False
                })

        return relevant_passages

    def inject_distractors(
        self,
        relevant_docs: List[Dict],
        quality_tier: str,
        k: int = 5
    ) -> List[Dict]:
        """
        Inject distractors based on quality tier

        Args:
            relevant_docs: List of relevant documents
            quality_tier: 'high', 'medium', or 'low'
            k: Total number of documents to return

        Returns:
            Mixed list of relevant and distractor documents
        """
        if self.passages is None:
            self.load_data()

        # Get ratios from config
        tier_config = self.config['retrieval']['quality_tiers'][quality_tier]
        relevant_ratio = tier_config['relevant_ratio']

        # Calculate how many of each type we need
        num_relevant = int(k * relevant_ratio)
        num_distractors = k - num_relevant

        # Sample relevant documents
        if len(relevant_docs) >= num_relevant:
            sampled_relevant = random.sample(relevant_docs, num_relevant)
        else:
            # Use all available relevant docs
            sampled_relevant = relevant_docs
            # Adjust distractor count
            num_distractors = k - len(sampled_relevant)

        # Sample distractors from passage pool
        distractors = []
        if num_distractors > 0:
            # Exclude relevant passage IDs
            relevant_ids = {d['id'] for d in relevant_docs}
            available_passages = [p for p in self.passages if p['id'] not in relevant_ids]

            distractor_passages = random.sample(
                available_passages,
                min(num_distractors, len(available_passages))
            )

            for passage in distractor_passages:
                distractors.append({
                    "id": passage['id'],
                    "score": 0.0,
                    "text": f"{passage['title']}\n{passage['text']}",
                    "title": passage['title'],
                    "is_distractor": True
                })

        # Combine and shuffle
        mixed_docs = sampled_relevant + distractors
        random.shuffle(mixed_docs)

        return mixed_docs

    def retrieve_with_quality_control(
        self,
        query: str,
        quality_tier: str = 'high',
        k: int = 5
    ) -> Tuple[List[Dict], Dict]:
        """
        Retrieve documents with controlled quality degradation

        Args:
            query: Query string (question text)
            quality_tier: 'high', 'medium', or 'low'
            k: Number of documents to retrieve

        Returns:
            (documents, metadata)
        """
        # Get relevant documents using mapping
        relevant_docs = self.get_relevant_passages(query)

        if not relevant_docs:
            print(f"⚠️  No relevant passages found for: {query}")
            print(f"   This question may not be answerable with the current corpus")

        # Inject distractors based on quality tier
        mixed_docs = self.inject_distractors(relevant_docs, quality_tier, k=k)

        # Prepare metadata
        metadata = {
            "query": query,
            "quality_tier": quality_tier,
            "method": "targeted",
            "num_relevant": len([d for d in mixed_docs if not d.get('is_distractor', False)]),
            "num_distractors": len([d for d in mixed_docs if d.get('is_distractor', False)]),
            "total_docs": len(mixed_docs)
        }

        return mixed_docs, metadata


def test_targeted_retriever(config):
    """Test the targeted retriever"""
    print("Testing Targeted Retriever...")

    retriever = TargetedRetriever(config)

    # Test question (from filtered set)
    test_query = "when was puerto rico added to the usa"

    # Test all quality tiers
    for tier in ['high', 'medium', 'low']:
        print(f"\n{'='*60}")
        print(f"Quality Tier: {tier.upper()}")
        print('='*60)

        docs, metadata = retriever.retrieve_with_quality_control(
            test_query,
            quality_tier=tier,
            k=5
        )

        print(f"Query: {test_query}")
        print(f"Retrieved: {metadata['total_docs']} docs "
              f"({metadata['num_relevant']} relevant, {metadata['num_distractors']} distractors)")

        for i, doc in enumerate(docs):
            print(f"\n{i+1}. {'[DISTRACTOR]' if doc.get('is_distractor') else '[RELEVANT]'}")
            print(f"   Title: {doc.get('title', 'N/A')}")
            print(f"   Text: {doc['text'][:150]}...")


if __name__ == "__main__":
    import yaml

    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    test_targeted_retriever(config)
