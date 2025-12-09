"""
Main RAG Pipeline: Orchestrates the entire experiment
Retrieves contexts -> Generates answers -> Detects hallucinations
"""

import json
import yaml
from pathlib import Path
import sys
from tqdm import tqdm
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.retriever import RAGRetriever
from src.generation.answer_generator import AnswerGenerator
from src.detection.ragas_detector import RAGASDetector
from src.detection.nli_detector import NLIDetector
from src.detection.lexical_detector import LexicalDetector

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_questions(config):
    """Load questions from dataset"""
    data_path = Path(config['paths']['raw_data']) / "nq_questions.jsonl"

    questions = []
    with open(data_path, 'r') as f:
        for line in f:
            questions.append(json.loads(line))

    return questions

def run_retrieval_phase(config, questions, quality_tier):
    """
    Run retrieval for all questions at a given quality tier

    Args:
        config: Configuration dict
        questions: List of question dicts
        quality_tier: 'high', 'medium', or 'low'

    Returns:
        List of retrieved contexts with metadata
    """
    print(f"\n{'='*60}")
    print(f"Retrieval Phase - Quality Tier: {quality_tier.upper()}")
    print('='*60)

    retriever = RAGRetriever(config)

    retrievals = []
    for question in tqdm(questions, desc=f"Retrieving ({quality_tier})"):
        docs, metadata = retriever.retrieve_with_quality_control(
            question['question'],
            quality_tier=quality_tier,
            method='dense',  # Can switch to 'bm25'
            k=5
        )

        retrievals.append({
            'question': question['question'],
            'ground_truth': question.get('answer', ''),
            'retrieved_docs': docs,
            'metadata': metadata
        })

    return retrievals

def run_generation_phase(config, retrievals):
    """
    Generate answers for all retrieved contexts

    Args:
        config: Configuration dict
        retrievals: List of retrieval results

    Returns:
        List of generated answers with contexts
    """
    print(f"\n{'='*60}")
    print("Answer Generation Phase")
    print('='*60)

    generator = AnswerGenerator(config)

    results = []
    for retrieval in tqdm(retrievals, desc="Generating answers"):
        # Generate answer
        answer_result = generator.generate_answer(
            retrieval['question'],
            retrieval['retrieved_docs']
        )

        # Combine with retrieval data
        result = {
            'question': retrieval['question'],
            'ground_truth': retrieval['ground_truth'],
            'context': [doc['text'] for doc in retrieval['retrieved_docs']],
            'answer': answer_result['answer'],
            'quality_tier': retrieval['metadata']['quality_tier'],
            'num_relevant': retrieval['metadata']['num_relevant'],
            'num_distractors': retrieval['metadata']['num_distractors']
        }

        results.append(result)

    return results

def run_detection_phase(config, qa_results):
    """
    Run all hallucination detectors

    Args:
        config: Configuration dict
        qa_results: List of QA results from generation phase

    Returns:
        Combined detection results
    """
    print(f"\n{'='*60}")
    print("Hallucination Detection Phase")
    print('='*60)

    # Initialize detectors
    print("\nInitializing detectors...")
    ragas_detector = RAGASDetector(config)
    nli_detector = NLIDetector(config)
    lexical_detector = LexicalDetector(config)

    all_results = []

    # Run each detector
    print("\n1. Running RAGAS detector...")
    try:
        ragas_results = ragas_detector.batch_detect(qa_results)
    except Exception as e:
        print(f"Warning: RAGAS detection failed: {e}")
        ragas_results = [{'is_hallucinated': None} for _ in qa_results]

    print("\n2. Running NLI detector...")
    try:
        nli_results = nli_detector.batch_detect(qa_results)
    except Exception as e:
        print(f"Warning: NLI detection failed: {e}")
        nli_results = [{'is_hallucinated': None} for _ in qa_results]

    print("\n3. Running Lexical Overlap detector...")
    try:
        lexical_results = lexical_detector.batch_detect(qa_results)
    except Exception as e:
        print(f"Warning: Lexical detection failed: {e}")
        lexical_results = [{'is_hallucinated': None} for _ in qa_results]

    # Combine results
    for i, qa in enumerate(qa_results):
        combined = {
            'question': qa['question'],
            'ground_truth': qa['ground_truth'],
            'answer': qa['answer'],
            'quality_tier': qa['quality_tier'],
            'num_relevant': qa['num_relevant'],
            'num_distractors': qa['num_distractors'],

            # RAGAS
            'ragas_hallucinated': ragas_results[i].get('is_hallucinated'),
            'ragas_faithfulness': ragas_results[i].get('ragas_faithfulness'),

            # NLI
            'nli_hallucinated': nli_results[i].get('is_hallucinated'),
            'nli_entailment_prob': nli_results[i].get('nli_entailment_prob'),

            # Lexical
            'lexical_hallucinated': lexical_results[i].get('is_hallucinated'),
            'lexical_overlap': lexical_results[i].get('lexical_combined_overlap'),
        }

        all_results.append(combined)

    return all_results

def label_ground_truth(results):
    """
    Label results with ground truth hallucination labels

    Args:
        results: List of detection results

    Returns:
        Results with ground truth labels added
    """
    print("\nLabeling ground truth...")

    for result in results:
        # Simple heuristic: compare answer to ground truth
        # In practice, this should be more sophisticated or manual
        answer_lower = result['answer'].lower()
        ground_truth_lower = str(result['ground_truth']).lower()

        # Check if ground truth appears in answer
        is_faithful = ground_truth_lower in answer_lower or answer_lower in ground_truth_lower

        result['ground_truth_hallucinated'] = not is_faithful

    return results

def save_results(config, results, quality_tier):
    """Save results to disk"""
    output_dir = Path(config['paths']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"results_{quality_tier}.jsonl"

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"\nSaved results to {output_path}")

    # Also save as CSV for easy viewing
    df = pd.DataFrame(results)
    csv_path = output_dir / f"results_{quality_tier}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

def run_full_pipeline(config, quality_tier, num_questions=100):
    """
    Run the complete RAG pipeline for a given quality tier

    Args:
        config: Configuration dict
        quality_tier: 'high', 'medium', or 'low'
        num_questions: Number of questions to process
    """
    print("\n" + "="*60)
    print(f"RUNNING FULL PIPELINE - QUALITY TIER: {quality_tier.upper()}")
    print("="*60)

    # Load questions
    print("\nLoading questions...")
    all_questions = load_questions(config)
    questions = all_questions[:num_questions]
    print(f"Processing {len(questions)} questions")

    # Phase 1: Retrieval
    retrievals = run_retrieval_phase(config, questions, quality_tier)

    # Phase 2: Generation
    qa_results = run_generation_phase(config, retrievals)

    # Phase 3: Detection
    detection_results = run_detection_phase(config, qa_results)

    # Phase 4: Ground truth labeling
    labeled_results = label_ground_truth(detection_results)

    # Save results
    save_results(config, labeled_results, quality_tier)

    return labeled_results

def main():
    """Main function"""
    config = load_config()

    print("="*60)
    print("RAG HALLUCINATION DETECTION PIPELINE")
    print("="*60)

    # Process each quality tier
    quality_tiers = ['high', 'medium', 'low']
    all_results = {}

    # Adjust number based on resources
    num_questions = 50  # Start small for testing

    for tier in quality_tiers:
        results = run_full_pipeline(config, tier, num_questions)
        all_results[tier] = results

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)

    # Print summary
    for tier, results in all_results.items():
        print(f"\n{tier.upper()} Quality Tier:")
        print(f"  Total samples: {len(results)}")

        # Count hallucinations detected by each method
        if results:
            ragas_count = sum(1 for r in results if r.get('ragas_hallucinated'))
            nli_count = sum(1 for r in results if r.get('nli_hallucinated'))
            lexical_count = sum(1 for r in results if r.get('lexical_hallucinated'))
            gt_count = sum(1 for r in results if r.get('ground_truth_hallucinated'))

            print(f"  Ground truth hallucinations: {gt_count}")
            print(f"  RAGAS detected: {ragas_count}")
            print(f"  NLI detected: {nli_count}")
            print(f"  Lexical detected: {lexical_count}")

if __name__ == "__main__":
    main()
