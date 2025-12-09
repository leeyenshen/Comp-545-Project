"""
Week 3: NLI-based Hallucination Detection
Uses Natural Language Inference models to check if answer is entailed by context
"""

import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict
import numpy as np

class NLIDetector:
    """
    NLI-based hallucination detector using entailment models
    """

    def __init__(self, config):
        self.config = config
        self.model_name = config['detection']['nli']['model_name']
        self.threshold = config['detection']['nli']['threshold']
        self.pipeline = None

    def load_model(self):
        """Load NLI model"""
        print(f"Loading NLI model: {self.model_name}")

        self.pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            device=0 if torch.cuda.is_available() else -1
        )

        print("NLI model loaded successfully")

    def check_entailment(
        self,
        premise: str,
        hypothesis: str
    ) -> Dict:
        """
        Check if hypothesis is entailed by premise

        Args:
            premise: Context text (premise)
            hypothesis: Generated answer (hypothesis)

        Returns:
            Dictionary with entailment score and label
        """
        if self.pipeline is None:
            self.load_model()

        # Format input for NLI
        # Some models expect specific format
        text = f"{premise} [SEP] {hypothesis}"

        # Get prediction
        result = self.pipeline(text)[0]

        # Extract label and score
        label = result['label'].lower()
        score = result['score']

        # Map to entailment probability
        if 'entailment' in label:
            entailment_prob = score
        elif 'contradiction' in label:
            entailment_prob = 0.0
        else:  # neutral
            entailment_prob = 0.5

        return {
            'label': label,
            'score': score,
            'entailment_probability': entailment_prob
        }

    def detect_hallucination(
        self,
        context: str,
        answer: str
    ) -> Dict:
        """
        Detect if answer is hallucinated based on context

        Args:
            context: Retrieved context
            answer: Generated answer

        Returns:
            Dictionary with detection results
        """
        # Check entailment
        entailment_result = self.check_entailment(context, answer)

        # Classify as hallucination if not entailed
        is_hallucinated = entailment_result['entailment_probability'] < self.threshold

        return {
            'entailment_probability': entailment_result['entailment_probability'],
            'entailment_label': entailment_result['label'],
            'is_hallucinated': is_hallucinated,
            'confidence': entailment_result['score']
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
                'nli_entailment_prob': detection_result['entailment_probability'],
                'nli_label': detection_result['entailment_label'],
                'is_hallucinated': detection_result['is_hallucinated'],
                'nli_confidence': detection_result['confidence']
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
        Check entailment against multiple context documents

        Args:
            contexts: List of context documents
            answer: Generated answer
            aggregation: How to aggregate scores ('max', 'mean', 'any')

        Returns:
            Aggregated detection result
        """
        entailment_probs = []

        for context in contexts:
            result = self.check_entailment(context, answer)
            entailment_probs.append(result['entailment_probability'])

        # Aggregate
        if aggregation == 'max':
            final_prob = max(entailment_probs)
        elif aggregation == 'mean':
            final_prob = np.mean(entailment_probs)
        elif aggregation == 'any':
            # If any context entails, consider it entailed
            final_prob = max(entailment_probs)
        else:
            final_prob = np.mean(entailment_probs)

        is_hallucinated = final_prob < self.threshold

        return {
            'entailment_probability': final_prob,
            'individual_probs': entailment_probs,
            'is_hallucinated': is_hallucinated
        }


def test_nli_detector(config):
    """Test NLI detector"""
    print("Testing NLI Detector...")

    detector = NLIDetector(config)

    # Test cases
    test_cases = [
        {
            'question': 'What is machine learning?',
            'context': 'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
            'answer': 'Machine learning is a subset of AI that learns from data.',
            'expected': False  # Not hallucinated
        },
        {
            'question': 'Who invented the telephone?',
            'context': 'Alexander Graham Bell invented the first practical telephone in 1876.',
            'answer': 'Thomas Edison invented the telephone.',
            'expected': True  # Hallucinated
        },
        {
            'question': 'What is the capital of France?',
            'context': 'Paris is the capital and most populous city of France.',
            'answer': 'London is the capital of France.',
            'expected': True  # Hallucinated
        }
    ]

    print("\nRunning detection on test cases...\n")

    for i, test in enumerate(test_cases):
        result = detector.detect_hallucination(test['context'], test['answer'])

        print(f"{'='*60}")
        print(f"Test Case {i+1}")
        print(f"Question: {test['question']}")
        print(f"Answer: {test['answer']}")
        print(f"Entailment Prob: {result['entailment_probability']:.3f}")
        print(f"Label: {result['entailment_label']}")
        print(f"Hallucinated: {result['is_hallucinated']}")
        print(f"Expected: {test['expected']}")
        print(f"Correct: {result['is_hallucinated'] == test['expected']}")
        print('='*60 + '\n')


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    test_nli_detector(config)
