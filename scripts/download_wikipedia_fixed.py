"""
Fixed Wikipedia Download Script
Downloads real Wikipedia passages using reliable methods
"""

import json
from pathlib import Path
from tqdm import tqdm
import yaml

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def method_1_wiki_dpr():
    """
    Method 1: Use wiki_dpr dataset (DPR preprocessed passages)
    This is the most reliable for retrieval tasks
    """
    print("\n" + "="*60)
    print("METHOD 1: Downloading wiki_dpr dataset")
    print("="*60)

    try:
        from datasets import load_dataset

        print("Loading wiki_dpr (DPR Wikipedia passages)...")
        print("This may take a few minutes on first download...")

        # Load the psgs_w100 split (100-word passages from Wikipedia)
        dataset = load_dataset(
            "facebook/dpr-ctx_encoder-multiset-base",
            split="train",
            trust_remote_code=True
        )

        print(f"✓ Successfully loaded {len(dataset)} passages")
        return dataset, "wiki_dpr"

    except Exception as e:
        print(f"✗ wiki_dpr failed: {e}")
        return None, None

def method_2_wikipedia_simple():
    """
    Method 2: Use Simple English Wikipedia
    Smaller and more reliable than full Wikipedia
    """
    print("\n" + "="*60)
    print("METHOD 2: Downloading Simple English Wikipedia")
    print("="*60)

    try:
        from datasets import load_dataset

        print("Loading Simple English Wikipedia...")

        # Load simple Wikipedia
        dataset = load_dataset(
            "wikipedia",
            "20220301.simple",
            split="train",
            trust_remote_code=True
        )

        print(f"✓ Successfully loaded {len(dataset)} articles")
        return dataset, "simple_wikipedia"

    except Exception as e:
        print(f"✗ Simple Wikipedia failed: {e}")
        return None, None

def method_3_wikipedia_en_sample():
    """
    Method 3: Use English Wikipedia with sampling
    """
    print("\n" + "="*60)
    print("METHOD 3: Downloading English Wikipedia (sample)")
    print("="*60)

    try:
        from datasets import load_dataset

        print("Loading English Wikipedia sample...")

        # Load streaming to avoid downloading everything
        dataset = load_dataset(
            "wikipedia",
            "20220301.en",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        # Take first 10000
        passages = []
        for i, item in enumerate(tqdm(dataset, total=10000, desc="Sampling")):
            if i >= 10000:
                break
            passages.append(item)

        print(f"✓ Successfully sampled {len(passages)} articles")
        return passages, "wikipedia_en"

    except Exception as e:
        print(f"✗ English Wikipedia failed: {e}")
        return None, None

def method_4_squad_context():
    """
    Method 4: Use SQuAD contexts as passages
    """
    print("\n" + "="*60)
    print("METHOD 4: Using SQuAD contexts as passages")
    print("="*60)

    try:
        from datasets import load_dataset

        print("Loading SQuAD dataset contexts...")

        dataset = load_dataset("squad", split="train")

        # Extract unique contexts
        unique_contexts = {}
        for item in tqdm(dataset, desc="Extracting contexts"):
            context = item['context']
            title = item['title']
            if context not in unique_contexts:
                unique_contexts[context] = title

        # Convert to passages
        passages = [
            {'text': context, 'title': title}
            for context, title in unique_contexts.items()
        ]

        print(f"✓ Successfully extracted {len(passages)} unique contexts")
        return passages, "squad"

    except Exception as e:
        print(f"✗ SQuAD failed: {e}")
        return None, None

def method_5_bookcorpus():
    """
    Method 5: Use BookCorpus
    """
    print("\n" + "="*60)
    print("METHOD 5: Downloading BookCorpus samples")
    print("="*60)

    try:
        from datasets import load_dataset

        print("Loading BookCorpus...")

        dataset = load_dataset("bookcorpus", split="train", streaming=True)

        passages = []
        for i, item in enumerate(tqdm(dataset, total=10000, desc="Sampling")):
            if i >= 10000:
                break
            passages.append({
                'text': item['text'],
                'title': f"Book Passage {i}"
            })

        print(f"✓ Successfully sampled {len(passages)} passages")
        return passages, "bookcorpus"

    except Exception as e:
        print(f"✗ BookCorpus failed: {e}")
        return None, None

def process_and_save(data, source_type, config, max_passages=10000):
    """
    Process downloaded data and save to JSONL
    """
    print(f"\nProcessing {source_type} data...")

    wiki_passages = []

    if source_type == "wiki_dpr":
        # DPR format
        for i, item in enumerate(tqdm(data.select(range(min(max_passages, len(data)))),
                                      desc="Processing")):
            wiki_passages.append({
                "id": str(item.get('id', f"wiki_{i}")),
                "title": item.get('title', 'Unknown'),
                "text": item.get('text', ''),
                "full_text": item.get('text', '')
            })

    elif source_type == "simple_wikipedia":
        # Simple Wikipedia format
        sample_size = min(max_passages, len(data))
        for i, article in enumerate(tqdm(data.select(range(sample_size)),
                                         desc="Processing")):
            text = article.get('text', '')[:1000]
            wiki_passages.append({
                "id": f"wiki_{i}",
                "title": article.get('title', 'Unknown'),
                "text": text,
                "full_text": text
            })

    elif source_type in ["wikipedia_en", "squad", "bookcorpus"]:
        # Already processed
        for i, item in enumerate(data[:max_passages]):
            wiki_passages.append({
                "id": f"wiki_{i}",
                "title": item.get('title', 'Unknown'),
                "text": item.get('text', ''),
                "full_text": item.get('text', '')
            })

    # Save to file
    output_path = Path(config['paths']['raw_data']) / "wikipedia_passages.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for passage in wiki_passages:
            f.write(json.dumps(passage) + '\n')

    print(f"✓ Saved {len(wiki_passages)} passages to {output_path}")

    # Print sample
    if wiki_passages:
        print("\nSample passage:")
        print(f"  Title: {wiki_passages[0]['title']}")
        print(f"  Text: {wiki_passages[0]['text'][:200]}...")

    return wiki_passages

def main():
    """Main function to try all methods"""
    config = load_config()

    print("="*60)
    print("WIKIPEDIA PASSAGES DOWNLOAD - MULTI-METHOD APPROACH")
    print("="*60)
    print("\nTrying multiple methods to download Wikipedia passages...")

    # Try each method in order
    methods = [
        method_4_squad_context,      # Most reliable - SQuAD contexts
        method_2_wikipedia_simple,   # Simple Wikipedia
        method_3_wikipedia_en_sample, # English Wikipedia sample
        method_1_wiki_dpr,           # DPR dataset
        method_5_bookcorpus,         # BookCorpus
    ]

    for method in methods:
        try:
            data, source_type = method()

            if data is not None:
                # Process and save
                passages = process_and_save(data, source_type, config)

                print("\n" + "="*60)
                print("✓ SUCCESS!")
                print("="*60)
                print(f"Source: {source_type}")
                print(f"Passages: {len(passages)}")
                print(f"Saved to: data/raw/wikipedia_passages.jsonl")
                print("="*60)
                return True

        except Exception as e:
            print(f"✗ Method failed with error: {e}")
            continue

    # If all methods fail, create synthetic data
    print("\n" + "="*60)
    print("⚠️  ALL DOWNLOAD METHODS FAILED")
    print("="*60)
    print("Creating synthetic passages as fallback...")

    from scripts.download_datasets import create_synthetic_passages
    passages = create_synthetic_passages(config)

    print(f"✓ Created {len(passages)} synthetic passages")
    return False

if __name__ == "__main__":
    success = main()

    if success:
        print("\n✅ Wikipedia passages successfully downloaded!")
    else:
        print("\n⚠️  Using synthetic passages (download failed)")

    print("\nNext step: python scripts/02_build_bm25_index.py")
