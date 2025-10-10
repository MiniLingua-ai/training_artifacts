import numpy as np
import pandas as pd
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import os

def extract_letter_answer(text, level='l2'):
    """Extract letter-based answers (A-G)"""
    if level == 'l1':
        pattern = r"answer is \(?([A-G])\)?"
        match = re.search(pattern, text)
        return match.group(1) if match else None
    elif level == 'l2':
        pattern = r"answer is \(?([A-G])\)?"
        match = re.search(pattern, text)
        return match.group(1) if match else extract_letter_again(text)

def extract_letter_again(text):
    """Fallback extraction for letter answers"""
    match = re.search(r'.*[aA]nswer:\s*([A-G])', text)
    return match.group(1) if match else extract_letter_final(text)

def extract_letter_final(text):
    """Final fallback for letter answers"""
    pattern = r"\b([A-G])\b(?!.*\b[A-G]\b)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None

def extract_number_answer(text, level='l2'):
    """Extract number-based answers (1-7)"""
    if level == 'l1':
        pattern = r"answer is \(?([1-7])\)?"
        match = re.search(pattern, text)
        return match.group(1) if match else None
    elif level == 'l2':
        pattern = r"answer is \(?([1-7])\)?"
        match = re.search(pattern, text)
        return match.group(1) if match else extract_number_again(text)

def extract_number_again(text):
    """Fallback extraction for number answers"""
    match = re.search(r'.*[aA]nswer:\s*([1-7])', text)
    return match.group(1) if match else extract_number_final(text)

def extract_number_final(text):
    """Final fallback for number answers"""
    pattern = r"\b([1-7])\b(?!.*\b[1-7]\b)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None

def extract_category_answer(text, answers):
    """Extract category-based answers (actual text answers)"""
    # Normalize text
    text = text.strip().lower()
    
    # Try fallback: find the last category mentioned in text
    for answer in answers:
        if answer.lower() in text:
            return answer.lower()
    
    return "N/A"

def get_language_configs(answer_type):
    """Get language configurations based on answer type"""
    # All answer types use the same language configurations for SIB
    return [
        ('cs', 'ces_Latn'), ('de', 'deu_Latn'), ('el', 'ell_Grek'), ('es', 'spa_Latn'),
        ('fi', 'fin_Latn'), ('fr', 'fra_Latn'), ('it', 'ita_Latn'), ('nl', 'nld_Latn'),
        ('pt', 'por_Latn'), ('sv', 'swe_Latn'), ('bg', 'bul_Cyrl'), ('pl', 'pol_Latn'), ('en', 'eng_Latn')
    ]

def format_example_letter(sample):
    """Format example for letter-based answers"""
    prompt = f"""Given the text below, choose the most appropriate topic from the given topics.
Finish your answer with "the answer is: X" where X is the correct letter choice.
{sample['text']}
A. science/technology
B. travel
C. politics
D. sports
E. health
F. entertainment
G. geography
"""
    return prompt

def format_example_number(sample):
    """Format example for number-based answers"""
    prompt = f"""Given the text below, choose the most appropriate topic from the given topics.
Finish your answer with "the answer is: X" where X is the correct number choice.
{sample['text']}
1. science/technology
2. travel
3. politics
4. sports
5. health
6. entertainment
7. geography
"""
    return prompt

def format_example_answer(sample):
    """Format example for answer-based responses"""
    prompt = f"""Given the text below, choose the most appropriate topic from the given categories. Read the text and return the correct topic.
{sample['text']}
- science/technology
- travel
- politics
- sports
- health
- entertainment
- geography
"""
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Unified SIB evaluation script with different answer formats.")
    parser.add_argument('--model_name', type=str, required=True, help='Path or identifier of the model to load.')
    parser.add_argument('--model', type=str, required=True, help='Model alias or custom identifier.')
    parser.add_argument('--answer_type', type=str, required=True, choices=['letter', 'number', 'answer'], 
                       help='Type of answer format: letter (A-G), number (1-7), or answer (actual text)')
    parser.add_argument('--base_path', type=str, required=True, help='Base directory for input/output paths.')

    args = parser.parse_args()

    # Build output directory based on answer type
    if args.answer_type == "letter":
        output_dir = os.path.join(args.base_path, "sib", args.model)
    elif args.answer_type == "number":
        output_dir = os.path.join(args.base_path, "sib_num", args.model)
    elif args.answer_type == "answer":
        output_dir = os.path.join(args.base_path, "sib_answer", args.model)
    
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    llm = LLM(model=args.model_name, dtype="float16")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    sampling_params = SamplingParams(temperature=0, max_tokens=200)

    # Define valid categories
    categories = [
        "science/technology", "travel", "politics", "sports", "health", 
        "entertainment", "geography"
    ]
    
    # Set up choices and mappings based on answer type
    if args.answer_type == "letter":
        choices = ["A", "B", "C", "D", "E", "F", "G"]
        category_labels = {
            "science/technology": "A",
            "travel": "B",
            "politics": "C",
            "sports": "D",
            "health": "E",
            "entertainment": "F",
            "geography": "G"
        }
    elif args.answer_type == "number":
        choices = ["1", "2", "3", "4", "5", "6", "7"]
        nums = {cat: str(i + 1) for i, cat in enumerate(categories)}
    else:  # answer type
        categories_lower = [c.lower() for c in categories]

    # Get language configurations
    langs = get_language_configs(args.answer_type)

    # Process each language
    for lang_config in langs:
        lang_code, hf_lang = lang_config

        prompts = []
        ground_truths = []
        predicted_answers = []
        model_outputs = []

        dataset = load_dataset('Davlan/sib200', hf_lang)

        for sample in tqdm(dataset['test'], desc=f"Processing {lang_code}"):
            # Format prompt based on answer type
            if args.answer_type == "letter":
                prompt = format_example_letter(sample)
            elif args.answer_type == "number":
                prompt = format_example_number(sample)
            elif args.answer_type == "answer":
                prompt = format_example_answer(sample)

            conversation = [{"role": "user", "content": prompt}]
            full_prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )

            prompts.append(full_prompt)

            # Set ground truth based on answer type
            if args.answer_type == "letter":
                ground_truths.append(category_labels[sample["category"].strip().lower()])
            elif args.answer_type == "number":
                ground_truths.append(nums[sample["category"]])
            elif args.answer_type == "answer":
                ground_truths.append(sample["category"].strip().lower())

        # Clean prompts and generate outputs
        prompts = [instr.replace("<|end_of_sequence|>", "") for instr in prompts]
        outputs = llm.generate(prompts, sampling_params)
        output_texts = [output.outputs[0].text for output in outputs]

        # Extract predictions based on answer type
        for i, text in enumerate(output_texts):
            if args.answer_type == "letter":
                pred = extract_letter_answer(text, level='l2')
                if pred not in choices:
                    pred = "N/A"
            elif args.answer_type == "number":
                pred = extract_number_answer(text, level='l2')
                if pred not in choices:
                    pred = "N/A"
            elif args.answer_type == "answer":
                pred = extract_category_answer(text, categories)
            
            predicted_answers.append(pred)
            model_outputs.append(text)

        # Evaluate
        if args.answer_type in ["letter", "number"]:
            correct = sum([1 for gt, pred in zip(ground_truths, predicted_answers) if gt == pred])
        elif args.answer_type == "answer":
            correct = sum([1 for gt, pred in zip(ground_truths, predicted_answers) if gt.strip().lower() == pred.strip().lower()])
        
        total = len(ground_truths)
        acc = correct / total

        # Save results
        with open(f'{output_dir}/sib_{lang_code}.txt', 'w') as fw:
            fw.write(str(acc))

        df = pd.DataFrame({
            'prompts': prompts,
            'answers': ground_truths,
            'predictions': predicted_answers,
            'model_output': model_outputs
        })
        df.to_csv(f'{output_dir}/sib_{lang_code}.tsv', sep='\t', index=False)

        print(f"Language: {lang_code}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
