#!/usr/bin/env python3
"""
Test RAGAS and NLI detectors to diagnose issues
"""

import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent))

def load_config():
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Test data
test_qa = [{
    'question': 'What is the capital of France?',
    'context': ['Paris is the capital and most populous city of France.'],
    'answer': 'Paris is the capital of France.',
    'ground_truth': 'Paris'
}]

print("="*60)
print("TESTING HALLUCINATION DETECTORS")
print("="*60)

config = load_config()

# Test 1: RAGAS (SKIPPED)
print("\n" + "="*60)
print("TEST 1: RAGAS Detector (SKIPPED)")
print("="*60)
print("Skipping RAGAS test as requested")

# Test 2: NLI
print("\n" + "="*60)
print("TEST 2: NLI Detector")
print("="*60)

try:
    from src.detection.nli_detector import NLIDetector

    print("✓ NLI module imported successfully")

    detector = NLIDetector(config)
    print("✓ NLI detector initialized")

    print("\nRunning detection...")
    results = detector.batch_detect(test_qa)

    print("✓ NLI detection completed!")
    print(f"Result: {results}")

except ImportError as e:
    print(f"✗ Import Error: {e}")

except Exception as e:
    print(f"✗ NLI Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Lexical (should work)
print("\n" + "="*60)
print("TEST 3: Lexical Detector (sanity check)")
print("="*60)

try:
    from src.detection.lexical_detector import LexicalDetector

    print("✓ Lexical module imported successfully")

    detector = LexicalDetector(config)
    print("✓ Lexical detector initialized")

    print("\nRunning detection...")
    results = detector.batch_detect(test_qa)

    print("✓ Lexical detection completed!")
    print(f"Result: {results}")

except Exception as e:
    print(f"✗ Lexical Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TESTING COMPLETE")
print("="*60)
