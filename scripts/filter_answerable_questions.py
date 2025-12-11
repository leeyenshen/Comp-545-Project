"""
Filter NQ questions to only those answerable by the available Wikipedia passages.

This ensures that the retrieval quality tiers (high/medium/low) have actual relevant
documents to work with, preventing all queries from resulting in abstentions.
"""

import json
import yaml
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_questions(config):
    """Load NQ questions"""
    data_path = Path(config['paths']['raw_data']) / "nq_questions.jsonl"

    questions = []
    with open(data_path, 'r') as f:
        for line in f:
            questions.append(json.loads(line))

    return questions


def load_passages(config):
    """Load Wikipedia passages"""
    passages_path = Path(config['paths']['raw_data']) / "wikipedia_passages.jsonl"

    passages = []
    with open(passages_path, 'r') as f:
        for line in tqdm(f, desc="Loading passages"):
            passages.append(json.loads(line))

    return passages


def extract_key_terms(question_text):
    """
    Extract key terms from a question for relevance checking.

    Args:
        question_text: Question string

    Returns:
        List of key terms (lowercase)
    """
    import re

    # Remove question words
    question_words = ['who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose', 'whom']
    text_lower = question_text.lower()

    for qw in question_words:
        text_lower = re.sub(r'\b' + qw + r'\b', '', text_lower)

    # Remove common stop words
    stop_words = ['is', 'are', 'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'by', 'with', 'did', 'do', 'does']
    for sw in stop_words:
        text_lower = re.sub(r'\b' + sw + r'\b', '', text_lower)

    # Extract remaining words (length > 2)
    words = re.findall(r'\b\w{3,}\b', text_lower)

    return words


def is_answerable(question, passages, require_question_relevance=True):
    """
    Check if a question is answerable by the available passages.

    A question is answerable if:
    1. At least one answer string appears in at least one passage
    2. (Optional) The passage also contains key terms from the question

    Args:
        question: Question dict with 'answer' field (list of acceptable answers)
        passages: List of passage dicts with 'text' and 'title' fields
        require_question_relevance: If True, also check if question terms appear in passage

    Returns:
        (is_answerable, matching_passage_ids): Boolean and list of passage IDs containing answer
    """
    answers = question.get('answer', [])
    if not answers:
        return False, []

    # Extract key terms from question for relevance check
    question_text = question.get('question', '')
    question_terms = extract_key_terms(question_text) if require_question_relevance else []

    matching_passage_ids = []

    # Check each answer variant
    for answer in answers:
        answer_lower = answer.lower().strip()

        # Skip very short answers (like single digits) that are too ambiguous
        # unless we're not requiring question relevance
        if require_question_relevance and len(answer_lower) < 4 and not answer_lower.isalpha():
            # For short numeric answers, we MUST have question context
            min_term_matches = max(2, len(question_terms) // 2)
        else:
            min_term_matches = 1 if require_question_relevance else 0

        # Search in passages
        for passage in passages:
            # Combine title and text for searching
            passage_text = f"{passage.get('title', '')} {passage.get('text', '')}".lower()

            # Check if answer appears in passage
            if answer_lower not in passage_text:
                continue

            # If requiring question relevance, check if enough question terms appear
            if require_question_relevance and min_term_matches > 0:
                term_matches = sum(1 for term in question_terms if term in passage_text)

                if term_matches < min_term_matches:
                    # Answer appears but passage is not about the question topic
                    continue

            # Both answer and question context match
            matching_passage_ids.append(passage['id'])
            return True, matching_passage_ids

    return False, []


def filter_answerable_questions(config):
    """
    Filter questions to only those answerable by available passages.

    Returns:
        List of answerable questions with metadata
    """
    print("Loading data...")
    questions = load_questions(config)
    passages = load_passages(config)

    print(f"\nTotal questions: {len(questions)}")
    print(f"Total passages: {len(passages)}")

    print("\nFiltering answerable questions...")
    answerable_questions = []
    question_to_passages = {}

    for question in tqdm(questions, desc="Filtering"):
        is_ans, matching_ids = is_answerable(question, passages)

        if is_ans:
            answerable_questions.append(question)
            question_to_passages[question['question']] = matching_ids

    print(f"\nAnswerable questions: {len(answerable_questions)}")
    print(f"Percentage: {100 * len(answerable_questions) / len(questions):.1f}%")

    return answerable_questions, question_to_passages


def save_filtered_questions(config, questions, question_to_passages):
    """Save filtered questions to disk"""
    output_dir = Path(config['paths']['raw_data'])

    # Save filtered questions
    output_path = output_dir / "nq_questions_filtered.jsonl"
    with open(output_path, 'w') as f:
        for question in questions:
            f.write(json.dumps(question) + '\n')

    print(f"\nSaved {len(questions)} filtered questions to {output_path}")

    # Save mapping (for analysis)
    mapping_path = output_dir / "question_passage_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(question_to_passages, f, indent=2)

    print(f"Saved question-passage mapping to {mapping_path}")

    # Print statistics
    print("\n" + "="*60)
    print("FILTERING STATISTICS")
    print("="*60)
    print(f"Original questions: {1000}")  # From config
    print(f"Answerable questions: {len(questions)}")
    print(f"Questions with answers in corpus: {100 * len(questions) / 1000:.1f}%")

    # Sample some answerable questions
    print("\n" + "="*60)
    print("SAMPLE ANSWERABLE QUESTIONS")
    print("="*60)
    for i, q in enumerate(questions[:5]):
        print(f"\n{i+1}. Question: {q['question']}")
        print(f"   Answer: {q['answer']}")
        matching_passages = question_to_passages.get(q['question'], [])
        print(f"   Found in {len(matching_passages)} passage(s): {matching_passages[:3]}")


def main():
    """Main function"""
    config = load_config()

    print("="*60)
    print("FILTERING ANSWERABLE QUESTIONS")
    print("="*60)
    print("\nThis script filters NQ questions to only those answerable")
    print("by the available 10k Wikipedia passages.")
    print("\nThis ensures that retrieval quality tiers have actual")
    print("relevant documents, preventing abstention responses.")
    print("="*60)

    # Filter questions
    answerable_questions, question_to_passages = filter_answerable_questions(config)

    if len(answerable_questions) == 0:
        print("\n⚠️  WARNING: No answerable questions found!")
        print("This suggests a complete mismatch between questions and passages.")
        print("Please check your data sources.")
        return

    if len(answerable_questions) < 50:
        print(f"\n⚠️  WARNING: Only {len(answerable_questions)} answerable questions found.")
        print("This is quite low. Consider downloading a larger passage corpus.")

    # Save results
    save_filtered_questions(config, answerable_questions, question_to_passages)

    print("\n" + "="*60)
    print("FILTERING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Update pipeline to use 'nq_questions_filtered.jsonl'")
    print("2. Run pipeline with filtered questions")
    print("3. Verify that quality tiers now have relevant documents")


if __name__ == "__main__":
    main()
