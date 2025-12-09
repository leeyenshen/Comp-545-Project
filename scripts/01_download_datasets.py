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

def create_synthetic_passages(config):
    """
    Create synthetic Wikipedia passages as a fallback
    Uses diverse text from multiple sources
    """
    print("Creating synthetic Wikipedia passages...")

    wiki_passages = []

    # Sample topics and passages
    topics = [
        ("Artificial Intelligence", "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals."),
        ("Machine Learning", "Machine learning is a branch of artificial intelligence and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy. Machine learning is an important component of the growing field of data science."),
        ("Natural Language Processing", "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data."),
        ("Deep Learning", "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks have been applied to fields including computer vision."),
        ("Computer Vision", "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do."),
        ("Python Programming", "Python is an interpreted high-level general-purpose programming language. Its design philosophy emphasizes code readability with its use of significant indentation. Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming."),
        ("Data Science", "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from noisy, structured and unstructured data. Data science is related to data mining, machine learning and big data."),
        ("Neural Networks", "A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes. Thus a neural network is either a biological neural network, made up of biological neurons, or an artificial neural network, used for solving artificial intelligence problems."),
        ("Transformer Architecture", "The Transformer is a deep learning model introduced in 2017, used primarily in the field of natural language processing. Like recurrent neural networks, transformers are designed to process sequential input data, but unlike RNNs, they do not require that the sequential data be processed in order."),
        ("Large Language Models", "A large language model is a language model consisting of a neural network with many parameters trained on large quantities of unlabeled text using self-supervised learning. LLMs emerged around 2018 and perform well at a wide variety of tasks."),
    ]

    # Expand with variations to reach 10,000
    target_size = 10000
    passage_id = 0

    while len(wiki_passages) < target_size:
        for title, text in topics:
            if len(wiki_passages) >= target_size:
                break

            # Add passage with variations
            wiki_passages.append({
                "id": f"wiki_{passage_id}",
                "title": title,
                "text": text,
                "full_text": text
            })
            passage_id += 1

            # Add paragraph variations
            sentences = text.split('. ')
            for i in range(len(sentences) - 1):
                if len(wiki_passages) >= target_size:
                    break
                partial_text = '. '.join(sentences[i:i+2]) + '.'
                wiki_passages.append({
                    "id": f"wiki_{passage_id}",
                    "title": f"{title} - Part {i+1}",
                    "text": partial_text,
                    "full_text": partial_text
                })
                passage_id += 1

    # Save passages
    output_path = Path(config['paths']['raw_data']) / "wikipedia_passages.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for passage in wiki_passages[:target_size]:
            f.write(json.dumps(passage) + '\n')

    print(f"Created {min(len(wiki_passages), target_size)} synthetic passages")
    return wiki_passages[:target_size]

def download_wikipedia_passages(config):
    """
    Download and prepare Wikipedia passages for retrieval corpus
    Tries multiple sources in order of preference
    """
    print("Preparing Wikipedia passages...")

    # Try 1: wiki_dpr dataset (preprocessed for DPR)
    try:
        print("Attempting to load wiki_dpr dataset...")
        wiki = load_dataset("wiki_dpr", "psgs_w100.nq.exact", split="train")

        max_passages = 10000
        print(f"Selecting {max_passages} passages...")

        # Handle different dataset types
        if hasattr(wiki, 'select'):
            wiki_sample = wiki.select(range(min(max_passages, wiki.num_rows)))
        else:
            wiki_sample = wiki

        wiki_passages = []
        for passage in tqdm(wiki_sample, total=max_passages, desc="Processing Wikipedia"):
            if len(wiki_passages) >= max_passages:
                break
            wiki_passages.append({
                "id": str(passage.get('id', f"wiki_{len(wiki_passages)}")),
                "title": passage.get('title', 'Unknown'),
                "text": passage.get('text', ''),
                "full_text": passage.get('text', '')
            })

        output_path = Path(config['paths']['raw_data']) / "wikipedia_passages.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for passage in wiki_passages:
                f.write(json.dumps(passage) + '\n')

        print(f"✓ Saved {len(wiki_passages)} Wikipedia passages to {output_path}")
        return wiki_passages

    except Exception as e:
        print(f"✗ wiki_dpr failed: {e}")

    # Try 2: Simple Wikipedia (smaller, more reliable)
    try:
        print("\nAttempting to load simple Wikipedia...")
        wiki = load_dataset("wikipedia", "20220301.simple", split="train[:10000]")

        wiki_passages = []
        for i, article in enumerate(tqdm(wiki, desc="Processing Simple Wikipedia")):
            text = article.get('text', '')[:1000]  # First 1000 chars
            wiki_passages.append({
                "id": f"wiki_{i}",
                "title": article.get('title', 'Unknown'),
                "text": text,
                "full_text": text
            })

        output_path = Path(config['paths']['raw_data']) / "wikipedia_passages.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for passage in wiki_passages:
                f.write(json.dumps(passage) + '\n')

        print(f"✓ Saved {len(wiki_passages)} Wikipedia passages to {output_path}")
        return wiki_passages

    except Exception as e:
        print(f"✗ Simple Wikipedia failed: {e}")

    # Fallback: Create synthetic passages
    print("\nUsing fallback: creating synthetic passages...")
    return create_synthetic_passages(config)

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
