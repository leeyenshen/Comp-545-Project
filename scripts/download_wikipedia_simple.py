"""
Simple and Reliable Wikipedia Download
Uses SQuAD contexts - guaranteed to work!
"""

import json
from pathlib import Path
from tqdm import tqdm
import yaml

def load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_squad_as_passages(config):
    """
    Download SQuAD dataset and use contexts as Wikipedia passages
    This is the most reliable method!
    """
    print("="*60)
    print("Downloading Wikipedia passages via SQuAD")
    print("="*60)

    from datasets import load_dataset

    # Load SQuAD - this always works
    print("\n1. Loading SQuAD dataset...")
    squad = load_dataset("squad", split="train")
    print(f"   ✓ Loaded {len(squad)} examples")

    # Extract unique contexts
    print("\n2. Extracting unique Wikipedia contexts...")
    unique_contexts = {}
    for item in tqdm(squad, desc="   Processing"):
        context = item['context']
        title = item['title']
        if context not in unique_contexts and len(context) > 50:
            unique_contexts[context] = title

    print(f"   ✓ Found {len(unique_contexts)} unique passages")

    # Create passages
    print("\n3. Creating passage documents...")
    wiki_passages = []
    for i, (context, title) in enumerate(tqdm(unique_contexts.items(),
                                              desc="   Formatting")):
        wiki_passages.append({
            "id": f"wiki_{i}",
            "title": title,
            "text": context,
            "full_text": context
        })

        # Stop at 10000 for efficiency
        if len(wiki_passages) >= 10000:
            break

    # Save to file
    print("\n4. Saving to file...")
    output_path = Path(config['paths']['raw_data']) / "wikipedia_passages.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for passage in wiki_passages:
            f.write(json.dumps(passage) + '\n')

    print(f"   ✓ Saved to {output_path}")

    # Show sample
    print("\n" + "="*60)
    print("SUCCESS! Sample passage:")
    print("="*60)
    print(f"Title: {wiki_passages[0]['title']}")
    print(f"Text: {wiki_passages[0]['text'][:300]}...")
    print("="*60)
    print(f"\nTotal passages: {len(wiki_passages)}")
    print(f"File: {output_path}")
    print("="*60)

    return wiki_passages

def main():
    config = load_config()

    print("\n🔍 Simple Wikipedia Download (via SQuAD)")
    print("This method is guaranteed to work!\n")

    passages = download_squad_as_passages(config)

    print("\n✅ COMPLETE!")
    print(f"✓ Downloaded {len(passages)} Wikipedia passages")
    print("✓ These are real Wikipedia contexts from SQuAD")
    print("\nNext step:")
    print("  python scripts/02_build_bm25_index.py")

if __name__ == "__main__":
    main()
