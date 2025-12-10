"""
Re-run detectors with OpenAI API for RAGAS (FAST!)
"""

import json
import yaml
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.detection.ragas_detector import RAGASDetector
from src.detection.nli_detector import NLIDetector
from src.detection.lexical_detector import LexicalDetector

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_existing_results(results_path):
    """Load existing results from JSONL file"""
    results = []
    with open(results_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def check_openai_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n❌ Error: OPENAI_API_KEY environment variable not set")
        print("\nPlease set it:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("\nOr run:")
        print("  ./setup_openai.sh")
        sys.exit(1)

    print(f"✓ OpenAI API key found: {api_key[:8]}...{api_key[-4:]}")
    return api_key

def run_detectors_with_openai(config, qa_results):
    """
    Run detectors with OpenAI API for RAGAS
    """
    print(f"\n{'='*60}")
    print("Running Detectors (OpenAI Mode)")
    print('='*60)
    print(f"Processing {len(qa_results)} QA pairs")

    # Initialize detectors
    print("\nInitializing detectors...")
    ragas_detector = RAGASDetector(config, use_openai=True)  # Use OpenAI!
    nli_detector = NLIDetector(config)
    lexical_detector = LexicalDetector(config)

    # Run each detector
    print("\n1. Running RAGAS detector (with OpenAI API)...")
    try:
        ragas_results = ragas_detector.batch_detect(qa_results)
        print("✓ RAGAS completed")
    except Exception as e:
        print(f"✗ RAGAS failed: {e}")
        import traceback
        traceback.print_exc()
        ragas_results = [{'is_hallucinated': None, 'ragas_faithfulness': None} for _ in qa_results]

    print("\n2. Running NLI detector...")
    try:
        nli_results = nli_detector.batch_detect(qa_results)
        print("✓ NLI completed")
    except Exception as e:
        print(f"✗ NLI failed: {e}")
        import traceback
        traceback.print_exc()
        nli_results = [{'is_hallucinated': None, 'nli_entailment_prob': None} for _ in qa_results]

    print("\n3. Running Lexical detector...")
    try:
        lexical_results = lexical_detector.batch_detect(qa_results)
        print("✓ Lexical completed")
    except Exception as e:
        print(f"✗ Lexical failed: {e}")
        import traceback
        traceback.print_exc()
        lexical_results = [{'is_hallucinated': None, 'lexical_combined_overlap': None} for _ in qa_results]

    # Combine with existing data
    updated_results = []
    for i, qa in enumerate(qa_results):
        # Keep original data
        result = qa.copy()

        # Update with detector results
        result.update({
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
            'ground_truth_hallucinated': qa.get('ground_truth_hallucinated', determine_ground_truth(qa))
        })

        updated_results.append(result)

    return updated_results

def determine_ground_truth(qa):
    """Determine if answer is hallucinated based on ground truth"""
    answer = qa['answer'].lower()
    ground_truth = str(qa.get('ground_truth', '')).lower()

    # Check for "don't know" patterns
    if any(phrase in answer for phrase in ["don't know", "do not know", "not mentioned", "no information"]):
        return True

    # Check if ground truth appears in answer
    if ground_truth and ground_truth in answer:
        return False

    return True

def save_results(results, output_path):
    """Save updated results"""
    output_path = Path(output_path)

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy types to native Python types"""
        import numpy as np
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    # Convert all results
    results = [convert_to_native(result) for result in results]

    # Save as JSONL
    with open(output_path.with_suffix('.jsonl'), 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Save as CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path.with_suffix('.csv'), index=False)

    print(f"\n✓ Updated results saved:")
    print(f"  - {output_path.with_suffix('.jsonl')}")
    print(f"  - {output_path.with_suffix('.csv')}")

def main():
    """Main function"""
    config = load_config()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     RE-RUN DETECTORS WITH OPENAI API                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Check OpenAI API key
    check_openai_key()

    print("\nThis will:")
    print("  ✓ Load existing QA results")
    print("  ✓ Run RAGAS with OpenAI API (GPT-3.5-turbo)")
    print("  ✓ Run NLI and Lexical detectors locally")
    print("  ✓ Update results with all detector scores")
    print("\n⚡ Much faster: ~5-10 minutes (vs 45-60 min with local)")
    print("💰 Cost: ~$0.10-0.50 for full experiment")
    print()

    results_dir = Path(config['paths']['results_dir'])

    # Process each quality tier
    quality_tiers = ['high', 'medium', 'low']

    for tier in quality_tiers:
        results_file = results_dir / f"results_{tier}.jsonl"

        if not results_file.exists():
            print(f"\n⚠️  Skipping {tier}: {results_file} not found")
            continue

        print(f"\n{'#'*60}")
        print(f"# Processing: {tier.upper()} quality tier")
        print(f"{'#'*60}")

        # Load existing results
        print(f"\nLoading {results_file}...")
        qa_results = load_existing_results(results_file)
        print(f"Loaded {len(qa_results)} QA pairs")

        # Convert to expected format if needed
        formatted_qa = []
        for qa in qa_results:
            formatted = {
                'question': qa['question'],
                'answer': qa['answer'],
                'context': qa.get('context', []),
                'ground_truth': qa.get('ground_truth', ''),
                'quality_tier': qa.get('quality_tier', tier),
                'num_relevant': qa.get('num_relevant', 0),
                'num_distractors': qa.get('num_distractors', 0)
            }
            formatted_qa.append(formatted)

        # Run detectors with OpenAI
        updated_results = run_detectors_with_openai(config, formatted_qa)

        # Save updated results
        output_path = results_dir / f"results_{tier}"
        save_results(updated_results, output_path)

    print("\n" + "="*60)
    print("ALL DETECTORS COMPLETE!")
    print("="*60)
    print(f"\nUpdated results saved to: {results_dir}")
    print("\nNext step: Run visualizations")
    print("  python scripts/05_create_visualizations.py")

if __name__ == "__main__":
    main()
