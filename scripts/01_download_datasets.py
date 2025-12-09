"""
Week 1: Data Download and Preparation
Downloads and processes NaturalQuestions and/or MuSiQue datasets
"""

import os
import json
import yaml
from datasets import load_dataset
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_natural_questions(config):
    """Download and process NaturalQuestions dataset"""
    print("Downloading NaturalQuestions dataset...")

    # Load dataset
    dataset = load_dataset("nq_open")

    # Extract subset
    subset_size = config['dataset']['subset_size']
    split = config['dataset']['split']

    sample = dataset[split].select(range(min(subset_size, len(dataset[split]))))

    # Process and save
    processed_data = []
    for item in tqdm(sample, desc="Processing NQ"):
        processed_data.append({
            "question": item['question'],
            "answer": item['answer'],
            "dataset": "natural_questions"
        })

    # Save to disk
    output_path = Path(config['paths']['raw_data']) / "nq_questions.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')

    print(f"Saved {len(processed_data)} questions to {output_path}")
    return processed_data

def download_musique(config):
    """Download and process MuSiQue dataset"""
    print("Downloading MuSiQue dataset...")

    try:
        # MuSiQue might need manual download
        # This is a placeholder for the actual implementation
        print("Note: MuSiQue may require manual download from:")
        print("https://github.com/StonyBrookNLP/musique")

        # For now, return empty list
        # TODO: Implement after manual download
        return []

    except Exception as e:
        print(f"Error downloading MuSiQue: {e}")
        return []

def download_wikipedia_passages(config):
    """
    Download and prepare Wikipedia passages for retrieval corpus
    Note: This is a simplified version. Full Wikipedia processing requires
    downloading the dump and using WikiExtractor.
    """
    print("Preparing Wikipedia passages...")
    print("Note: For full Wikipedia, download from:")
    print("https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2")
    print("\nFor this demo, we'll use a sample of Wikipedia from datasets library")

    try:
        # Use Wikipedia dataset from HuggingFace as a starting point
        wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

        # Take a sample
        wiki_passages = []
        max_passages = 10000  # Adjust based on resources

        for i, article in enumerate(tqdm(wiki, total=max_passages, desc="Loading Wikipedia")):
            if i >= max_passages:
                break

            # Split article into passages (simplified)
            text = article['text']
            title = article['title']

            # Store passage
            wiki_passages.append({
                "id": f"wiki_{i}",
                "title": title,
                "text": text[:1000],  # Truncate for simplicity
                "full_text": text
            })

        # Save passages
        output_path = Path(config['paths']['raw_data']) / "wikipedia_passages.jsonl"
        with open(output_path, 'w') as f:
            for passage in wiki_passages:
                f.write(json.dumps(passage) + '\n')

        print(f"Saved {len(wiki_passages)} Wikipedia passages to {output_path}")
        return wiki_passages

    except Exception as e:
        print(f"Error downloading Wikipedia: {e}")
        print("You may need to download Wikipedia dump manually")
        return []

def main():
    """Main function to orchestrate data downloading"""
    config = load_config()

    print("="*60)
    print("Week 1: Data Download and Preparation")
    print("="*60)

    # Download based on config
    dataset_name = config['dataset']['name']

    if dataset_name == "natural_questions":
        questions = download_natural_questions(config)
    elif dataset_name == "musique":
        questions = download_musique(config)
    else:
        print(f"Unknown dataset: {dataset_name}")
        return

    # Download Wikipedia passages
    passages = download_wikipedia_passages(config)

    print("\n" + "="*60)
    print("Data download complete!")
    print(f"Questions downloaded: {len(questions)}")
    print(f"Wikipedia passages downloaded: {len(passages)}")
    print("="*60)

if __name__ == "__main__":
    main()
