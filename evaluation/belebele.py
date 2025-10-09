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
    """Extract letter-based answers (A-D)"""
    if level == 'l1':
        pattern = r"answer is \(?([A-D])\)?"
        match = re.search(pattern, text)
        return match.group(1) if match else None
    elif level == 'l2':
        pattern = r"answer is \(?([A-D])\)?"
        match = re.search(pattern, text)
        return match.group(1) if match else extract_letter_again(text)

def extract_letter_again(text):
    """Fallback extraction for letter answers"""
    match = re.search(r'.*[aA]nswer:\s*([A-D])', text)
    return match.group(1) if match else extract_letter_final(text)

def extract_letter_final(text):
    """Final fallback for letter answers"""
    pattern = r"\b([A-D])\b(?!.*\b[A-D]\b)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None

def extract_number_answer(text, level='l2'):
    """Extract number-based answers (1-4)"""
    if level == 'l1':
        pattern = r"answer is \(?([1-4])\)?"
        match = re.search(pattern, text)
        return match.group(1) if match else None
    elif level == 'l2':
        pattern = r"answer is \(?([1-4])\)?"
        match = re.search(pattern, text)
        return match.group(1) if match else extract_number_again(text)

def extract_number_again(text):
    """Fallback extraction for number answers"""
    match = re.search(r'.*[aA]nswer:\s*([1-4])', text)
    return match.group(1) if match else extract_number_final(text)

def extract_number_final(text):
    """Final fallback for number answers"""
    pattern = r"\b([1-4])\b(?!.*\b[1-4]\b)"
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
    if answer_type == "letter":
        return [
            ('cs', 'ces_Latn'), ('de', 'deu_Latn'), ('el', 'ell_Grek'), ('es', 'spa_Latn'),
            ('fi', 'fin_Latn'), ('fr', 'fra_Latn'), ('it', 'ita_Latn'), ('nl', 'nld_Latn'),
            ('pt', 'por_Latn'), ('sv', 'swe_Latn'), ('bg', 'bul_Cyrl'), ('pl', 'pol_Latn'), ('en', 'eng_Latn')
        ]
    elif answer_type == "number":
        return [
            ('cs', 'ces_Latn'),
            ('de', 'deu_Latn'),
            ('el', 'ell_Grek'),
            ('es', 'spa_Latn'),
            ('fi', 'fin_Latn'),
            ('fr', 'fra_Latn'),
            ('it', 'ita_Latn'),
            ('nl', 'nld_Latn'),
            ('pt', 'por_Latn'),
            ('sv', 'swe_Latn'),
            ('bg', 'bul_Cyrl'),
            ('pl', 'pol_Latn'),
            ('en', 'eng_Latn')
        ]
    elif answer_type == "answer":
        return [
            ('cs', 'ces_Latn'),
            ('de', 'deu_Latn'),
            ('el', 'ell_Grek'),
            ('es', 'spa_Latn'),
            ('fi', 'fin_Latn'),
            ('fr', 'fra_Latn'),
            ('it', 'ita_Latn'),
            ('nl', 'nld_Latn'),
            ('pt', 'por_Latn'),
            ('sv', 'swe_Latn'),
            ('bg', 'bul_Cyrl'),
            ('pl', 'pol_Latn'),
            ('en', 'eng_Latn')
        ]

def format_example_letter(sample):
    """Format example for letter-based answers"""
    prompt = f"""The following is a multiple choice question about the text. Read the text and answer the question. Finish your answer with "the answer is (X)" where X is the correct letter choice.
{sample["flores_passage"]}
{sample["question"]}
A. {sample[f'mc_answer1']}
B. {sample[f'mc_answer2']}
C. {sample[f'mc_answer3']}
D. {sample[f'mc_answer4']}
"""
    return prompt

def format_example_number(sample):
    """Format example for number-based answers"""
    prompt = f"""The following is a multiple choice question about the text. Read the text and answer the question. Finish your answer with "the answer is (X)" where X is the correct number choice.
{sample["flores_passage"]}
{sample["question"]}
1. {sample[f'mc_answer1']}
2. {sample[f'mc_answer2']}
3. {sample[f'mc_answer3']}
4. {sample[f'mc_answer4']}
"""
    return prompt

def format_example_answer(sample):
    """Format example for answer-based responses"""
    prompt = f"""The following is a multiple choice question about the text. Read the text and return correct answer.
{sample["flores_passage"]}
{sample["question"]}
- {sample[f'mc_answer1']}
- {sample[f'mc_answer2']}
- {sample[f'mc_answer3']}
- {sample[f'mc_answer4']}
"""
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Unified Belebele evaluation script with different answer formats.")
    parser.add_argument('--model_name', type=str, required=True, help='Path or identifier of the model to load.')
    parser.add_argument('--model', type=str, required=True, help='Model alias or custom identifier.')
    parser.add_argument('--answer_type', type=str, required=True, choices=['letter', 'number', 'answer'], 
                       help='Type of answer format: letter (A-D), number (1-4), or answer (actual text)')
    args = parser.parse_args()

    # Set up output directory based on answer type
    if args.answer_type == "letter":
        output_dir = f"/scratch/cs/small_lm/eval_scripts/belebele_alt/{args.model}"
    elif args.answer_type == "number":
        output_dir = f"/scratch/cs/small_lm/eval_scripts/belebele_num_lang/{args.model}"
    elif args.answer_type == "answer":
        output_dir = f"/scratch/cs/small_lm/eval_scripts/belebele_answer_lang/{args.model}"
    
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    llm = LLM(model=args.model_name, dtype="float16")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    sampling_params = SamplingParams(temperature=0, max_tokens=200)

    # Set up choices based on answer type
    if args.answer_type == "letter":
        choices = ["A", "B", "C", "D"]
    else:
        choices = ["1", "2", "3", "4"]

    # Get language configurations
    langs = get_language_configs(args.answer_type)

    # Process each language
    for lang_config in langs:
        lang_code, hf_lang = lang_config

        prompts = []
        ground_truths = []
        predicted_answers = []
        model_outputs = []
        answers_list = []

        dataset = load_dataset('facebook/belebele', hf_lang, trust_remote_code=True)

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
                correct_letter = choices[int(sample["correct_answer_num"]) - 1]
                ground_truths.append(correct_letter)
            elif args.answer_type == "number":
                correct_number = choices[int(sample["correct_answer_num"]) - 1]
                ground_truths.append(correct_number)
            elif args.answer_type == "answer":
                ground_truths.append(sample[f'mc_answer{sample["correct_answer_num"]}'])
                answers_list.append([sample[f'mc_answer{i}'] for i in range(1, 5)])

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
                pred = extract_category_answer(text, answers_list[i])
            
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
        with open(f'{output_dir}/belebele_{lang_code}.txt', 'w') as fw:
            fw.write(str(acc))

        df = pd.DataFrame({
            'prompts': prompts,
            'answers': ground_truths,
            'predictions': predicted_answers,
            'model_output': model_outputs
        })
        df.to_csv(f'{output_dir}/belebele_{lang_code}.tsv', sep='\t', index=False)

        print(f"Language: {lang_code}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
