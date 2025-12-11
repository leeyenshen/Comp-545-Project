#!/bin/bash
# Quick test with just 1 question

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     SINGLE QUESTION TEST                                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Activate virtual environment
source venv/bin/activate

# Run for just 1 question
python3 << 'EOF'
import sys
import importlib.util
from pathlib import Path

# Load the pipeline module
spec = importlib.util.spec_from_file_location("pipeline", "scripts/04_run_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
sys.modules["pipeline"] = pipeline
spec.loader.exec_module(pipeline)

# Run for HIGH tier with just 1 question
config = pipeline.load_config()
print("Running test with 1 question...")
results = pipeline.run_full_pipeline(config, 'high', num_questions=1)

# Print detailed results
print("\n" + "="*60)
print("TEST RESULTS")
print("="*60)
if len(results) > 0:
    r = results[0]
    print(f"\nQuestion: {r['question']}")
    print(f"Ground Truth: {r['ground_truth']}")
    print(f"\nContext (first 200 chars): {str(r['context'][:1])[0:200] if r.get('context') else 'NO CONTEXT'}...")
    print(f"Number of context docs: {len(r.get('context', []))}")
    print(f"\nAnswer: {r['answer'][:200]}...")
    print(f"\nDetection Results:")
    print(f"  RAGAS hallucinated: {r.get('ragas_hallucinated')}")
    print(f"  NLI hallucinated: {r.get('nli_hallucinated')}")
    print(f"  Lexical hallucinated: {r.get('lexical_hallucinated')}")
    print(f"  Ground truth hallucinated: {r.get('ground_truth_hallucinated')}")
    print(f"\n✓ Test completed successfully!")
else:
    print("❌ No results generated")
EOF

echo ""
echo "Test complete!"
