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

    def __init__(self, config, use_openai=True):
        """
        Initialize RAGAS detector

        Args:
            config: Configuration dict
            use_openai: If True, use OpenAI API (fast). If False, use local models (slow).
        """
        self.config = config
        self.metrics_config = config['detection']['ragas']['metrics']
        self.use_openai = use_openai

        # Initialize models based on preference
        if use_openai:
            print("✓ Using OpenAI API for RAGAS (hybrid mode: OpenAI LLM + local embeddings)")
            self._init_openai_hybrid()
        else:
            print("Using local models for RAGAS (slow mode)")
            self._init_local_models()

        # Map metric names to RAGAS metrics
        self.metric_map = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'context_recall': context_recall
        }

    def _init_openai_hybrid(self):
        """Initialize OpenAI LLM with local embeddings (hybrid for RAGAS 0.4+)"""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            # Use OpenAI for LLM (fast)
            self.llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))

            # Use local HuggingFace embeddings (avoids OpenAI embed_query issue)
            self.embeddings = LangchainEmbeddingsWrapper(
                HFEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )

            print("✓ Hybrid models initialized (OpenAI LLM + local embeddings)")

        except Exception as e:
            print(f"Warning: Could not initialize hybrid models: {e}")
            print("Falling back to defaults")
            self.llm = None
            self.embeddings = None

    def _init_local_models(self):
        """Initialize local LLM and embeddings for RAGAS"""
        print("Initializing local models for RAGAS...")

        try:
            # Use a small local LLM for RAGAS with optimized settings for speed
            llm_pipeline = pipeline(
                "text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                max_new_tokens=50,   # Reduced for speed
                temperature=0.1,     # Lower = faster, more deterministic
                do_sample=False,     # Greedy = faster
                pad_token_id=50256
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
        # Add timeout and batch configuration
        if self.llm and self.embeddings:
            # Configure execution with longer timeout and smaller batches
            from ragas import RunConfig
            run_config = RunConfig(
                timeout=300.0,  # 5 minutes per item (increased from default 60s)
                max_workers=2,   # Reduce parallelism to avoid overload
                max_wait=1800.0  # 30 minute total timeout
            )

            results = evaluate(
                dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                run_config=run_config,
                raise_exceptions=False  # Continue on errors
            )
        else:
            # Fallback to default (may require OpenAI API key)
            results = evaluate(dataset, metrics=metrics)

        return results

    def classify_hallucination(
        self,
        faithfulness_score: float,
        answer: str = None,
        threshold: float = 0.5
    ) -> bool:
        """
        Classify if an answer is hallucinated based on faithfulness score

        Args:
            faithfulness_score: RAGAS faithfulness score
            answer: The generated answer text (optional, for abstention detection)
            threshold: Threshold for classification

        Returns:
            True if hallucinated, False otherwise
        """
        # Check if answer is an abstention (refusing to answer due to insufficient info)
        # Abstention is NOT a hallucination - it's the correct response to poor context
        if answer:
            abstention_phrases = [
                "i don't know",
                "i do not know",
                "don't know",
                "do not know",
                "cannot answer",
                "unable to answer",
                "insufficient information",
                "not mentioned in the context",
                "not mentioned in the provided context",
                "no information",
                "does not mention",
                "does not contain"
            ]

            answer_lower = answer.lower()
            if any(phrase in answer_lower for phrase in abstention_phrases):
                # Abstention is NOT a hallucination
                return False

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

        # Fix ground_truth format: convert list to string if needed
        ground_truths = []
        for qa in qa_pairs:
            gt = qa.get('ground_truth', '')
            # Convert list to string (take first element or join)
            if isinstance(gt, list):
                gt = gt[0] if gt else ''
            ground_truths.append(str(gt))

        # Run RAGAS
        ragas_results = self.detect(questions, contexts, answers, ground_truths)

        # RAGAS 0.4+ returns a Result object with .scores attribute
        # Convert to dict format
        if hasattr(ragas_results, 'scores'):
            # New RAGAS 0.4+ format
            scores_dict = ragas_results.scores
            results_df = pd.DataFrame(scores_dict)
        elif hasattr(ragas_results, 'to_pandas'):
            # Alternative: has to_pandas method
            results_df = ragas_results.to_pandas()
        elif isinstance(ragas_results, dict):
            # Already a dict
            results_df = pd.DataFrame(ragas_results)
        else:
            # Unknown format, create empty results
            print(f"Warning: Unknown RAGAS result format: {type(ragas_results)}")
            results_df = pd.DataFrame({
                'faithfulness': [0.5] * len(qa_pairs),
                'answer_relevancy': [0.5] * len(qa_pairs)
            })

        # Add hallucination predictions
        results = []
        for i, qa in enumerate(qa_pairs):
            faithfulness = results_df.loc[i, 'faithfulness'] if 'faithfulness' in results_df.columns else 0.5
            answer_relevancy = results_df.loc[i, 'answer_relevancy'] if 'answer_relevancy' in results_df.columns else None

            result = {
                'question': qa['question'],
                'answer': qa['answer'],
                'ragas_faithfulness': faithfulness,
                'ragas_answer_relevancy': answer_relevancy,
                'is_hallucinated': self.classify_hallucination(faithfulness, answer=qa['answer'])
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
