"""
Week 2: LLM Answer Generation
Generates answers using instruction-tuned LLMs (Mistral/Llama)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Tuple
import yaml
from pathlib import Path

class AnswerGenerator:
    """
    LLM-based answer generator for RAG
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None

    def load_model(self):
        """Load instruction-tuned LLM"""
        model_name = self.config['generation']['model_name']
        load_in_8bit = self.config['generation'].get('load_in_8bit', False)

        print(f"Loading model: {model_name}")

        # Configure quantization if needed
        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        else:
            quantization_config = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if not load_in_8bit else None,
            trust_remote_code=True
        )

        self.device = self.model.device
        print(f"Model loaded on device: {self.device}")

    def format_prompt(self, question: str, context: List[Dict]) -> str:
        """
        Format question and context into a prompt for the LLM

        Args:
            question: Question string
            context: List of retrieved documents

        Returns:
            Formatted prompt string
        """
        # Combine context documents
        context_text = "\n\n".join([
            f"Document {i+1}: {doc['text'][:500]}"  # Truncate long docs
            for i, doc in enumerate(context)
        ])

        # Format prompt (adjust based on model)
        model_name = self.config['generation']['model_name'].lower()

        if 'mistral' in model_name:
            # Mistral format
            prompt = f"""<s>[INST] Answer the following question based on the provided context. If the context doesn't contain the answer, say "I don't know."

Context:
{context_text}

Question: {question}

Answer: [/INST]"""

        elif 'llama' in model_name:
            # Llama-2 format
            prompt = f"""<s>[INST] <<SYS>>
You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain the answer, say "I don't know."
<</SYS>>

Context:
{context_text}

Question: {question} [/INST]"""

        else:
            # Generic format
            prompt = f"""Context:
{context_text}

Question: {question}

Answer:"""

        return prompt

    def generate_answer(
        self,
        question: str,
        context: List[Dict],
        **generation_kwargs
    ) -> Dict:
        """
        Generate answer given question and context

        Args:
            question: Question string
            context: List of retrieved documents
            **generation_kwargs: Additional generation parameters

        Returns:
            Dictionary with answer and metadata
        """
        if self.model is None:
            self.load_model()

        # Format prompt
        prompt = self.format_prompt(question, context)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        # Set generation parameters
        gen_config = {
            "max_new_tokens": self.config['generation'].get('max_new_tokens', 200),
            "temperature": self.config['generation'].get('temperature', 0.7),
            "top_p": self.config['generation'].get('top_p', 0.9),
            "do_sample": self.config['generation'].get('do_sample', True),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Override with provided kwargs
        gen_config.update(generation_kwargs)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_config)

        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer (remove prompt)
        answer = full_output[len(prompt):].strip()

        return {
            "question": question,
            "answer": answer,
            "prompt": prompt,
            "full_output": full_output
        }

    def generate_batch(
        self,
        questions_and_contexts: List[Tuple],
        **generation_kwargs
    ) -> List[Dict]:
        """
        Generate answers for a batch of questions

        Args:
            questions_and_contexts: List of (question, context) tuples

        Returns:
            List of answer dictionaries
        """
        results = []

        for question, context in questions_and_contexts:
            result = self.generate_answer(question, context, **generation_kwargs)
            results.append(result)

        return results


def test_generator(config):
    """Test the answer generator"""
    print("Testing Answer Generator...")

    generator = AnswerGenerator(config)

    # Test question and context
    test_question = "What is machine learning?"

    test_context = [
        {
            "text": "Machine learning is a subset of artificial intelligence that "
                    "enables systems to learn and improve from experience without being "
                    "explicitly programmed. It focuses on developing computer programs "
                    "that can access data and use it to learn for themselves."
        },
        {
            "text": "The main types of machine learning are supervised learning, "
                    "unsupervised learning, and reinforcement learning."
        }
    ]

    print(f"\nQuestion: {test_question}")
    print(f"Context documents: {len(test_context)}")

    # Generate answer
    result = generator.generate_answer(test_question, test_context)

    print(f"\nGenerated Answer:")
    print(result['answer'])
    print("\n" + "="*60)


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    test_generator(config)
