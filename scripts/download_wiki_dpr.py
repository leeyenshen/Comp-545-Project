"""
Download diverse Wikipedia passages from wiki_dpr dataset
Creates a 100k passage corpus with good coverage
"""

import json
from pathlib import Path
from tqdm import tqdm
import yaml
from datasets import load_dataset
from collections import defaultdict

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_diverse_wiki_dpr(target_passages=100000):
    """
    Download diverse Wikipedia passages using legacy datasets

    Uses the legacy DPR Wikipedia passages dataset from Facebook
    """
    print(f"Downloading Wikipedia passages (targeting {target_passages:,})...")
    print("This may take 10-15 minutes depending on your internet connection.\n")

    try:
        # Use the legacy DPR wikipedia passages - this should work without loading scripts
        print("Loading Wikipedia passages from legacy DPR dataset...")
        print("Dataset: legacy/wikipedia-dpr")

        # Try using the streaming mode to avoid loading scripts
        dataset = load_dataset(
            "legacy/wikipedia-dpr",
            split="train",
            streaming=True
        )

        print("Streaming dataset loaded successfully!")
        print(f"Sampling {target_passages:,} passages...")

        passages = []
        title_count = defaultdict(int)

        # Process streamed data
        for i, article in enumerate(tqdm(dataset, total=target_passages, desc="Downloading passages")):
            if len(passages) >= target_passages:
                break

            # Extract title and text
            title = article.get('title', 'Unknown')
            text = article.get('text', '')

            if text.strip():
                passages.append({
                    'id': f"wiki_{len(passages)}",
                    'title': title,
                    'text': text.strip(),
                    'full_text': text.strip()
                })
                title_count[title] += 1

        print(f"\nCollected {len(passages):,} passages")
        print(f"Unique articles: {len(title_count):,}")
        print(f"\nTop 10 most common articles:")
        for title, count in sorted(title_count.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {title}: {count} passages")

        return passages

    except Exception as e:
        print(f"\n❌ Legacy DPR dataset failed: {e}")
        print("\nTrying alternative: Wikimedia/wikipedia...")

        try:
            # Try wikimedia/wikipedia which should be in parquet
            dataset = load_dataset(
                "wikimedia/wikipedia",
                "20231101.en",
                split="train",
                streaming=True
            )

            print("Wikimedia dataset loaded!")
            print(f"Sampling {target_passages:,} passages...")

            passages = []
            title_count = defaultdict(int)
            max_passage_length = 500

            for i, article in enumerate(tqdm(dataset, total=target_passages*2, desc="Processing")):
                if len(passages) >= target_passages:
                    break

                title = article.get('title', 'Unknown')
                text = article.get('text', '')

                # Split long articles
                if len(text) > max_passage_length:
                    chunks = [text[j:j+max_passage_length] for j in range(0, len(text), max_passage_length)]
                    for chunk in chunks[:5]:  # Max 5 passages per article
                        if len(passages) >= target_passages:
                            break
                        if chunk.strip():
                            passages.append({
                                'id': f"wiki_{len(passages)}",
                                'title': title,
                                'text': chunk.strip(),
                                'full_text': chunk.strip()
                            })
                            title_count[title] += 1
                else:
                    if text.strip():
                        passages.append({
                            'id': f"wiki_{len(passages)}",
                            'title': title,
                            'text': text.strip(),
                            'full_text': text.strip()
                        })
                        title_count[title] += 1

            print(f"\nCollected {len(passages):,} passages")
            print(f"Unique articles: {len(title_count):,}")

            return passages

        except Exception as e2:
            print(f"\n❌ Wikimedia dataset also failed: {e2}")
            print("\n⚠️  All HuggingFace options failed.")
            print("\nFinal fallback: Expanding current corpus...")

            # Last resort: just use what we have and warn the user
            print("\n" + "="*60)
            print("Using current 10k corpus (limited coverage)")
            print("="*60)
            print("\nThis will work but with reduced question coverage.")
            print("The pipeline will still run, but many questions may be unanswerable.")

            # Return None to signal we should use existing corpus
            return None

def save_passages(passages, config):
    """Save passages to disk"""
    output_path = Path(config['paths']['raw_data']) / "wikipedia_passages.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup old file if it exists
    if output_path.exists():
        backup_path = output_path.with_suffix('.jsonl.backup')
        print(f"\nBacking up old corpus to {backup_path}")
        output_path.rename(backup_path)

    print(f"Saving passages to {output_path}...")
    with open(output_path, 'w') as f:
        for passage in tqdm(passages, desc="Writing"):
            f.write(json.dumps(passage) + '\n')

    print(f"✓ Saved {len(passages):,} passages")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

def main():
    config = load_config()

    print("="*60)
    print("DOWNLOAD DIVERSE WIKIPEDIA CORPUS")
    print("="*60)
    print()

    # Download 100k diverse passages
    passages = download_diverse_wiki_dpr(target_passages=100000)

    # Check if download succeeded
    if passages is None:
        print("\n⚠️  Download failed. Using existing corpus.")
        print("The pipeline will work with your current 10k passages.")
        print("Results may have limited coverage, but detectors will still function.")
        print("\nYou can still run:")
        print("  ./run_pipeline_safe.sh")
        return

    # Save to disk
    save_passages(passages, config)

    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run: python scripts/03_build_faiss_index.py")
    print("  2. Run: ./run_pipeline_safe.sh")

if __name__ == "__main__":
    main()
