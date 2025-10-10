import torch
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
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        return match.group(1) if match else None
    elif level == 'l2':
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        return match.group(1) if match else extract_letter_again(text)

def extract_letter_again(text):
    """Fallback extraction for letter answers"""
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    return match.group(1) if match else extract_letter_final(text)

def extract_letter_final(text):
    """Final fallback for letter answers"""
    pattern = r"\b([A-J])\b(?!.*\b[A-J]\b)"
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

def get_language_configs():
    """Get language configurations"""
    return ['CS', 'DE', 'EL', 'ES', 'FI', 'FR', 'IT', 'NL', 'SV', 'BG', 'PL', 'PT-PT']

def get_mmlu_configs():
    """Get MMLU dataset configurations
    BG used by default and will be replaced with the language code.
    """
    return ['abstract_algebra_BG', 'anatomy_BG', 'astronomy_BG',
           'business_ethics_BG', 'clinical_knowledge_BG', 'college_biology_BG',
           'college_chemistry_BG', 'college_computer_science_BG', 'college_mathematics_BG',
           'college_medicine_BG', 'college_physics_BG', 'computer_security_BG',
           'conceptual_physics_BG', 'econometrics_BG', 'electrical_engineering_BG',
           'elementary_mathematics_BG', 'formal_logic_BG', 'global_facts_BG',
           'high_school_biology_BG', 'high_school_chemistry_BG',
           'high_school_computer_science_BG', 'high_school_european_history_BG',
           'high_school_geography_BG', 'high_school_government_and_politics_BG',
           'high_school_macroeconomics_BG', 'high_school_mathematics_BG',
           'high_school_microeconomics_BG', 'high_school_physics_BG',
           'high_school_psychology_BG', 'high_school_statistics_BG',
           'high_school_us_history_BG', 'high_school_world_history_BG',
           'human_aging_BG', 'human_sexuality_BG', 'international_law_BG',
           'jurisprudence_BG', 'logical_fallacies_BG', 'machine_learning_BG',
           'management_BG', 'marketing_BG', 'medical_genetics_BG', 'miscellaneous_BG',
           'moral_disputes_BG', 'moral_scenarios_BG', 'nutrition_BG', 'philosophy_BG',
           'prehistory_BG', 'professional_accounting_BG', 'professional_law_BG',
           'professional_medicine_BG', 'professional_psychology_BG', 'public_relations_BG',
           'security_studies_BG', 'sociology_BG', 'us_foreign_policy_BG', 'virology_BG', 'world_religions_BG']

def format_example_letter(sample, subject):
    """Format example for letter-based answers (A-D)"""
    prompt = f"""The following is a multiple choice question about {subject}. Answer the question. Finish your answer with "the answer is (X)" where X is the correct letter choice.
{sample["question"]}
A. {sample["choices"][0]}
B. {sample["choices"][1]}
C. {sample["choices"][2]}
D. {sample["choices"][3]}
"""
    return prompt

def format_example_number(sample, subject):
    """Format example for number-based answers (1-4)"""
    prompt = f"""The following is a multiple choice question about {subject}. Answer the question. Finish your answer with "the answer is (X)" where X is the correct number choice.
{sample["question"]}
1. {sample["choices"][0]}
2. {sample["choices"][1]}
3. {sample["choices"][2]}
4. {sample["choices"][3]}
"""
    return prompt

def format_example_answer(sample, subject):
    """Format example for answer-based responses"""
    prompt = f"""The following is a multiple choice question about {subject}. Answer the question. Return correct answer.
{sample["question"]}
- {sample["choices"][0]}
- {sample["choices"][1]}
- {sample["choices"][2]}
- {sample["choices"][3]}
"""
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Unified MMLU evaluation script with different answer formats.")
    parser.add_argument('--model_name', type=str, required=True, help='Path or identifier of the model to load.')
    parser.add_argument('--model', type=str, required=True, help='Model alias or custom identifier.')
    parser.add_argument('--answer_type', type=str, required=True, choices=['letter', 'number', 'answer'], 
                       help='Type of answer format: letter (A-D), number (1-4), or answer (actual text)')
    parser.add_argument('--base_path', type=str, required=True, help='Base directory for input/output paths.')

    args = parser.parse_args()

    # Build output directory based on answer type
    if args.answer_type == "letter":
        output_dir = os.path.join(args.base_path, "mmlu", args.model)
    elif args.answer_type == "number":
        output_dir = os.path.join(args.base_path, "mmlu_num", args.model)
    elif args.answer_type == "answer":
        output_dir = os.path.join(args.base_path, "mmlu_answer", args.model)
    
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

    # Get configurations
    langs = get_language_configs()
    configs = get_mmlu_configs()

    # Process each language
    for lang in langs:
        prompts = []
        ground_truths = []
        predicted_answers = []
        options = []

        for config in configs:
            config = config.replace('BG', lang)
            dataset = load_dataset('openGPT-X/mmlux', config, trust_remote_code=True)

            for sample in tqdm(dataset['test'], desc=f"Processing {config}"):
                subject = sample['id']
                subject = ' '.join(subject.split('/')[0].split('_'))

                # Format prompt based on answer type
                if args.answer_type == "letter":
                    test_prompt = format_example_letter(sample, subject)
                elif args.answer_type == "number":
                    test_prompt = format_example_number(sample, subject)
                elif args.answer_type == "answer":
                    test_prompt = format_example_answer(sample, subject)

                conversation = [{"role": "user", "content": test_prompt}]
                full_prompt = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False
                )
                full_prompt = full_prompt.replace("<|end_of_sequence|>", "")

                prompts.append(full_prompt)
                
                # Set ground truth based on answer type
                if args.answer_type == "letter":
                    ground_truths.append(choices[sample["answer"]])
                elif args.answer_type == "number":
                    ground_truths.append(choices[sample["answer"]])
                elif args.answer_type == "answer":
                    ground_truths.append(sample["choices"][int(sample["answer"])])
                    options.append(sample["choices"])

        # Generate outputs
        outputs = llm.generate(prompts, sampling_params)
        output_texts = [output.outputs[0].text for output in outputs]

        # Extract predictions based on answer type
        for i, text in enumerate(output_texts):
            if args.answer_type == "letter":
                pred = extract_letter_answer(text, level='l2')
                if pred is None:
                    pred = "N/A"
            elif args.answer_type == "number":
                pred = extract_number_answer(text, level='l2')
                if pred is None:
                    pred = "N/A"
            elif args.answer_type == "answer":
                pred = extract_category_answer(text, options[i])
            
            predicted_answers.append(pred)

        # Evaluate
        if args.answer_type in ["letter", "number"]:
            correct = sum([1 for gt, pred in zip(ground_truths, predicted_answers) if gt == pred])
        elif args.answer_type == "answer":
            correct = sum([1 for gt, pred in zip(ground_truths, predicted_answers) if gt.lower().strip() == pred.lower().strip()])
        
        total = len(ground_truths)
        acc = correct / total

        # Save results
        with open(os.path.join(output_dir, f"mmlu_{lang.lower()}.txt"), 'w') as fw:
            fw.write(str(acc))

        df = pd.DataFrame({
            'prompts': prompts,
            'answers': ground_truths,
            'predictions': predicted_answers,
            'model_output': output_texts
        })

        df.to_csv(os.path.join(output_dir, f"mmlu_{lang.lower()}.tsv"), sep='\t', index=False)
        
        print(f"Language: {lang}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
