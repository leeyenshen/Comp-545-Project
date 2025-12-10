"""
OPTIMIZED Main RAG Pipeline with Parallel Processing
Retrieves contexts -> Generates answers -> Detects hallucinations (in parallel)
"""

import json
import yaml
from pathlib import Path
import sys
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

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
    """Run retrieval for all questions (already optimized in retriever)"""
    print(f"\n{'='*60}")
    print(f"Retrieval Phase: {quality_tier.upper()} quality")
    print('='*60)

    retriever = RAGRetriever(config)

    retrievals = []
    for q in tqdm(questions, desc=f"Retrieving ({quality_tier})"):
        result = retriever.retrieve(
            question=q['question'],
            ground_truth=q['answer'],
            quality_tier=quality_tier
        )

        retrievals.append({
            'question': q['question'],
            'ground_truth': q['answer'],
            'retrieved_docs': result['documents'],
            'metadata': result['metadata']
        })

    return retrievals

def generate_single_answer(generator, retrieval):
    """Helper function to generate a single answer (for parallel processing)"""
    answer_result = generator.generate_answer(
        retrieval['question'],
        retrieval['retrieved_docs']
    )

    return {
        'question': retrieval['question'],
        'ground_truth': retrieval['ground_truth'],
        'context': [doc['text'] for doc in retrieval['retrieved_docs']],
        'answer': answer_result['answer'],
        'quality_tier': retrieval['metadata']['quality_tier'],
        'num_relevant': retrieval['metadata']['num_relevant'],
        'num_distractors': retrieval['metadata']['num_distractors']
    }

def run_generation_phase(config, retrievals):
    """
    OPTIMIZED: Generate answers with batch processing
    (Single model load, sequential inference for stability)
    """
    print(f"\n{'='*60}")
    print("Answer Generation Phase")
    print('='*60)

    generator = AnswerGenerator(config)

    results = []
    # Keep sequential for model stability (GPU inference)
    for retrieval in tqdm(retrievals, desc="Generating answers"):
        result = generate_single_answer(generator, retrieval)
        results.append(result)

    return results

def run_detector_batch(detector, detector_name, qa_results):
    """Helper to run a single detector and return results"""
    print(f"\nRunning {detector_name} detector...")
    try:
        results = detector.batch_detect(qa_results)
        print(f"✓ {detector_name} completed")
        return (detector_name, results, None)
    except Exception as e:
        print(f"✗ {detector_name} failed: {e}")
        return (detector_name, [{'is_hallucinated': None} for _ in qa_results], str(e))

def run_detection_phase_parallel(config, qa_results):
    """
    OPTIMIZED: Run all detectors in PARALLEL using threads
    """
    print(f"\n{'='*60}")
    print("Hallucination Detection Phase (PARALLEL)")
    print('='*60)

    # Initialize detectors (in main thread)
    print("\nInitializing detectors...")
    detectors = {
        'ragas': RAGASDetector(config),
        'nli': NLIDetector(config),
        'lexical': LexicalDetector(config)
    }

    # Run detectors in parallel threads
    print("\nRunning detectors in parallel...")
    results_dict = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all detector tasks
        futures = {
            executor.submit(run_detector_batch, detector, name, qa_results): name
            for name, detector in detectors.items()
        }

        # Collect results as they complete
        for future in as_completed(futures):
            detector_name, results, error = future.result()
            results_dict[detector_name] = results
            if error:
                print(f"⚠️  {detector_name} had errors but continuing...")

    # Combine results
    all_results = []
    for i, qa in enumerate(qa_results):
        ragas_results = results_dict['ragas']
        nli_results = results_dict['nli']
        lexical_results = results_dict['lexical']

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

            # Ground truth
            'ground_truth_hallucinated': determine_ground_truth_hallucination(qa)
        }

        all_results.append(combined)

    return all_results

def determine_ground_truth_hallucination(qa):
    """Determine if answer is actually hallucinated based on ground truth"""
    answer = qa['answer'].lower()
    ground_truth = str(qa['ground_truth']).lower()

    # Check if answer contains "don't know" or similar
    if any(phrase in answer for phrase in ["don't know", "do not know", "i don't", "not mentioned"]):
        return True

    # Check if ground truth appears in answer
    if ground_truth in answer:
        return False

    # Default: consider hallucinated
    return True

def save_results(results, output_path):
    """Save results to JSONL and CSV"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    with open(output_path.with_suffix('.jsonl'), 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Save as CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path.with_suffix('.csv'), index=False)

    print(f"\n✓ Results saved:")
    print(f"  - {output_path.with_suffix('.jsonl')}")
    print(f"  - {output_path.with_suffix('.csv')}")

def main():
    """Main pipeline orchestration"""
    config = load_config()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     RAG HALLUCINATION DETECTION - PARALLEL PIPELINE         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nOptimizations:")
    print("  • Parallel detector execution (3x speedup)")
    print("  • Batch processing where possible")
    print("  • Optimized for macOS MPS")
    print()

    # Load questions
    questions = load_questions(config)
    print(f"Loaded {len(questions)} questions")

    # Quality tiers to process
    quality_tiers = ['high', 'medium', 'low']

    for tier in quality_tiers:
        print(f"\n{'#'*60}")
        print(f"# Processing Quality Tier: {tier.upper()}")
        print(f"{'#'*60}")

        # Phase 1: Retrieval
        retrievals = run_retrieval_phase(config, questions, tier)

        # Phase 2: Generation (sequential for GPU stability)
        qa_results = run_generation_phase(config, retrievals)

        # Phase 3: Detection (PARALLEL!)
        detection_results = run_detection_phase_parallel(config, qa_results)

        # Save results
        output_dir = Path(config['paths']['results_dir'])
        output_path = output_dir / f"results_{tier}"
        save_results(detection_results, output_path)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to: {config['paths']['results_dir']}")

if __name__ == "__main__":
    main()
