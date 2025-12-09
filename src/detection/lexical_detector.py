"""
Week 3: Lexical Overlap-based Hallucination Detection
Simple detector based on token overlap between answer and context
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Set
import string

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class LexicalDetector:
    """
    Lexical overlap-based hallucination detector
    """

    def __init__(self, config):
        self.config = config
        self.threshold = config['detection']['lexical']['overlap_threshold']
        self.use_lemmatization = config['detection']['lexical'].get('use_lemmatization', True)

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer() if self.use_lemmatization else None

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize and normalize text

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize
        tokens = text.split()

        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words]

        # Lemmatize
        if self.use_lemmatization and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return tokens

    def extract_named_entities(self, text: str) -> Set[str]:
        """
        Extract potential named entities (capitalized words/phrases)

        Args:
            text: Input text

        Returns:
            Set of potential named entities
        """
        # Simple heuristic: find capitalized words
        entities = set()

        # Find sequences of capitalized words
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)

        entities.update(matches)

        return entities

    def compute_overlap(
        self,
        context: str,
        answer: str
    ) -> Dict:
        """
        Compute token overlap between context and answer

        Args:
            context: Context text
            answer: Generated answer

        Returns:
            Dictionary with overlap metrics
        """
        # Tokenize
        context_tokens = set(self.tokenize(context))
        answer_tokens = set(self.tokenize(answer))

        # Compute overlap
        if len(answer_tokens) == 0:
            token_overlap = 0.0
        else:
            token_overlap = len(context_tokens & answer_tokens) / len(answer_tokens)

        # Extract and compare named entities
        context_entities = self.extract_named_entities(context)
        answer_entities = self.extract_named_entities(answer)

        if len(answer_entities) == 0:
            entity_overlap = 1.0  # No entities to check
        else:
            entity_overlap = len(context_entities & answer_entities) / len(answer_entities)

        return {
            'token_overlap': token_overlap,
            'entity_overlap': entity_overlap,
            'context_tokens': len(context_tokens),
            'answer_tokens': len(answer_tokens),
            'shared_tokens': len(context_tokens & answer_tokens),
            'context_entities': list(context_entities),
            'answer_entities': list(answer_entities),
            'shared_entities': list(context_entities & answer_entities)
        }

    def detect_hallucination(
        self,
        context: str,
        answer: str
    ) -> Dict:
        """
        Detect if answer is hallucinated based on lexical overlap

        Args:
            context: Retrieved context
            answer: Generated answer

        Returns:
            Dictionary with detection results
        """
        # Compute overlap
        overlap_metrics = self.compute_overlap(context, answer)

        # Classify as hallucination if overlap is too low
        # Consider both token and entity overlap
        combined_score = (overlap_metrics['token_overlap'] + overlap_metrics['entity_overlap']) / 2

        is_hallucinated = combined_score < self.threshold

        return {
            'token_overlap': overlap_metrics['token_overlap'],
            'entity_overlap': overlap_metrics['entity_overlap'],
            'combined_overlap': combined_score,
            'is_hallucinated': is_hallucinated,
            'details': overlap_metrics
        }

    def batch_detect(
        self,
        qa_pairs: List[Dict]
    ) -> List[Dict]:
        """
        Detect hallucinations in a batch of QA pairs

        Args:
            qa_pairs: List of dictionaries with 'question', 'context', 'answer'

        Returns:
            List of detection results
        """
        results = []

        for qa in qa_pairs:
            # Combine context if it's a list
            if isinstance(qa['context'], list):
                context = " ".join(qa['context'])
            else:
                context = qa['context']

            # Detect hallucination
            detection_result = self.detect_hallucination(context, qa['answer'])

            # Combine with original data
            result = {
                'question': qa['question'],
                'answer': qa['answer'],
                'lexical_token_overlap': detection_result['token_overlap'],
                'lexical_entity_overlap': detection_result['entity_overlap'],
                'lexical_combined_overlap': detection_result['combined_overlap'],
                'is_hallucinated': detection_result['is_hallucinated']
            }
            results.append(result)

        return results

    def multi_context_detection(
        self,
        contexts: List[str],
        answer: str,
        aggregation: str = 'max'
    ) -> Dict:
        """
        Check overlap against multiple context documents

        Args:
            contexts: List of context documents
            answer: Generated answer
            aggregation: How to aggregate scores ('max', 'mean')

        Returns:
            Aggregated detection result
        """
        overlaps = []

        for context in contexts:
            result = self.detect_hallucination(context, answer)
            overlaps.append(result['combined_overlap'])

        # Aggregate
        if aggregation == 'max':
            final_overlap = max(overlaps)
        else:  # mean
            final_overlap = sum(overlaps) / len(overlaps)

        is_hallucinated = final_overlap < self.threshold

        return {
            'combined_overlap': final_overlap,
            'individual_overlaps': overlaps,
            'is_hallucinated': is_hallucinated
        }


def test_lexical_detector(config):
    """Test lexical detector"""
    print("Testing Lexical Overlap Detector...")

    detector = LexicalDetector(config)

    # Test cases
    test_cases = [
        {
            'question': 'What is machine learning?',
            'context': 'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
            'answer': 'Machine learning is a subset of AI that learns from data.',
            'expected': False  # Not hallucinated (high overlap)
        },
        {
            'question': 'Who invented the telephone?',
            'context': 'Alexander Graham Bell invented the first practical telephone in 1876.',
            'answer': 'Thomas Edison invented the telephone.',
            'expected': True  # Hallucinated (low overlap - wrong name)
        },
        {
            'question': 'What is the capital of France?',
            'context': 'Paris is the capital and most populous city of France.',
            'answer': 'The capital of France is Paris.',
            'expected': False  # Not hallucinated (high overlap)
        },
        {
            'question': 'What is quantum computing?',
            'context': 'Machine learning uses algorithms to find patterns in data.',
            'answer': 'Quantum computing uses qubits to perform calculations.',
            'expected': True  # Hallucinated (no overlap)
        }
    ]

    print("\nRunning detection on test cases...\n")

    for i, test in enumerate(test_cases):
        result = detector.detect_hallucination(test['context'], test['answer'])

        print(f"{'='*60}")
        print(f"Test Case {i+1}")
        print(f"Question: {test['question']}")
        print(f"Answer: {test['answer']}")
        print(f"Token Overlap: {result['token_overlap']:.3f}")
        print(f"Entity Overlap: {result['entity_overlap']:.3f}")
        print(f"Combined Overlap: {result['combined_overlap']:.3f}")
        print(f"Hallucinated: {result['is_hallucinated']}")
        print(f"Expected: {test['expected']}")
        print(f"Correct: {result['is_hallucinated'] == test['expected']}")

        if result['details']['answer_entities']:
            print(f"Answer Entities: {result['details']['answer_entities']}")
            print(f"Shared Entities: {result['details']['shared_entities']}")

        print('='*60 + '\n')


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    test_lexical_detector(config)
