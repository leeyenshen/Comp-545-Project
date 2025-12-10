"""
Week 3: RAGAS-based Hallucination Detection
Uses RAGAS framework for multi-faceted hallucination detection
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from typing import List, Dict
import pandas as pd
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import torch

class RAGASDetector:
    """
    RAGAS-based hallucination detector using local models
    """

    def __init__(self, config):
        self.config = config
        self.metrics_config = config['detection']['ragas']['metrics']

        # Initialize local models for RAGAS
        self._init_local_models()

        # Map metric names to RAGAS metrics
        self.metric_map = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'context_recall': context_recall
        }

    def _init_local_models(self):
        """Initialize local LLM and embeddings for RAGAS"""
        print("Initializing local models for RAGAS...")

        try:
            # Use a small local LLM for RAGAS
            # Use TinyLlama for speed
            llm_pipeline = pipeline(
                "text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                max_new_tokens=100,
                temperature=0.7
            )

            self.llm = LangchainLLMWrapper(HuggingFacePipeline(pipeline=llm_pipeline))

            # Use local embeddings
            self.embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )

            print("✓ Local models initialized for RAGAS")

        except Exception as e:
            print(f"Warning: Could not initialize local models: {e}")
            print("RAGAS will use defaults (may require OpenAI API key)")
            self.llm = None
            self.embeddings = None

    def prepare_data(
        self,
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: List[str] = None
    ) -> Dataset:
        """
        Prepare data in RAGAS format

        Args:
            questions: List of questions
            contexts: List of context lists (each context is a list of documents)
            answers: List of generated answers
            ground_truths: List of ground truth answers (optional)

        Returns:
            RAGAS Dataset
        """
        data = {
            'question': questions,
            'contexts': contexts,
            'answer': answers
        }

        if ground_truths:
            data['ground_truth'] = ground_truths

        return Dataset.from_dict(data)

    def detect(
        self,
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: List[str] = None
    ) -> Dict:
        """
        Run RAGAS detection

        Args:
            questions: List of questions
            contexts: List of context lists
            answers: List of generated answers
            ground_truths: List of ground truth answers (optional)

        Returns:
            Dictionary of RAGAS scores
        """
        # Prepare dataset
        dataset = self.prepare_data(questions, contexts, answers, ground_truths)

        # Select metrics
        metrics = [
            self.metric_map[metric_name]
            for metric_name in self.metrics_config
            if metric_name in self.metric_map
        ]

        # Evaluate with local models if available
        if self.llm and self.embeddings:
            results = evaluate(
                dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
        else:
            # Fallback to default (may require OpenAI API key)
            results = evaluate(dataset, metrics=metrics)

        return results

    def classify_hallucination(
        self,
        faithfulness_score: float,
        threshold: float = 0.5
    ) -> bool:
        """
        Classify if an answer is hallucinated based on faithfulness score

        Args:
            faithfulness_score: RAGAS faithfulness score
            threshold: Threshold for classification

        Returns:
            True if hallucinated, False otherwise
        """
        # Lower faithfulness = more hallucination
        return faithfulness_score < threshold

    def batch_detect(
        self,
        qa_pairs: List[Dict]
    ) -> List[Dict]:
        """
        Detect hallucinations in a batch of QA pairs

        Args:
            qa_pairs: List of dictionaries with 'question', 'context', 'answer', 'ground_truth'

        Returns:
            List of results with hallucination predictions
        """
        # Extract data
        questions = [qa['question'] for qa in qa_pairs]
        contexts = [qa['context'] for qa in qa_pairs]
        answers = [qa['answer'] for qa in qa_pairs]
        ground_truths = [qa.get('ground_truth', '') for qa in qa_pairs]

        # Run RAGAS
        ragas_results = self.detect(questions, contexts, answers, ground_truths)

        # Convert to DataFrame for easier manipulation
        results_df = pd.DataFrame(ragas_results)

        # Add hallucination predictions
        results = []
        for i, qa in enumerate(qa_pairs):
            result = {
                'question': qa['question'],
                'answer': qa['answer'],
                'ragas_faithfulness': results_df.loc[i, 'faithfulness'] if 'faithfulness' in results_df else None,
                'ragas_answer_relevancy': results_df.loc[i, 'answer_relevancy'] if 'answer_relevancy' in results_df else None,
                'is_hallucinated': self.classify_hallucination(
                    results_df.loc[i, 'faithfulness'] if 'faithfulness' in results_df else 0.5
                )
            }
            results.append(result)

        return results


def test_ragas_detector(config):
    """Test RAGAS detector"""
    print("Testing RAGAS Detector...")

    detector = RAGASDetector(config)

    # Test data
    test_data = [
        {
            'question': 'What is machine learning?',
            'context': [
                'Machine learning is a subset of AI that enables systems to learn from data.',
                'It uses algorithms to find patterns in data.'
            ],
            'answer': 'Machine learning is a subset of AI that enables systems to learn from data.',
            'ground_truth': 'Machine learning is a type of artificial intelligence.'
        },
        {
            'question': 'Who invented the telephone?',
            'context': [
                'Alexander Graham Bell was a Scottish-born inventor.',
                'Bell is credited with inventing the first practical telephone.'
            ],
            'answer': 'Thomas Edison invented the telephone.',  # Hallucination
            'ground_truth': 'Alexander Graham Bell invented the telephone.'
        }
    ]

    # Run detection
    results = detector.batch_detect(test_data)

    # Display results
    for i, result in enumerate(results):
        print(f"\n{'='*60}")
        print(f"Example {i+1}")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Faithfulness: {result['ragas_faithfulness']:.3f}")
        print(f"Hallucinated: {result['is_hallucinated']}")
        print('='*60)


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    test_ragas_detector(config)
