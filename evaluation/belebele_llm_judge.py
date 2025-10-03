import numpy as np
import pandas as pd
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import os

def extract_answer(text, level='l2'):
    if level == 'l1':
        pattern = r"answer is \(?([A-D])\)?"
        match = re.search(pattern, text)
        return match.group(1) if match else None
    elif level == 'l2':
        pattern = r"answer is \(?([A-D])\)?"
        match = re.search(pattern, text)
        return match.group(1) if match else extract_again(text)

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-D])', text)
    return match.group(1) if match else extract_final(text)

def extract_final(text):
    pattern = r"\b([A-D])\b(?!.*\b[A-D]\b)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None

parser = argparse.ArgumentParser(description="Load model and language setup.")
parser.add_argument('--model_name', type=str, required=True, help='Path or identifier of the model to load.')
parser.add_argument('--model', type=str, required=True, help='Model alias or custom identifier.')
args = parser.parse_args()

output_dir = f"/scratch/cs/small_lm/eval_scripts/belebele_llm_judge/{args.model}"
os.makedirs(output_dir, exist_ok=True)

llm = LLM(model=args.model_name, dtype="float16")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
sampling_params = SamplingParams(temperature=0, max_tokens=200)

choices = ["A", "B", "C", "D"]

langs = [
    ('cs', 'ces_Latn'), ('de', 'deu_Latn'), ('el', 'ell_Grek'), ('es', 'spa_Latn'),
    ('fi', 'fin_Latn'), ('fr', 'fra_Latn'), ('it', 'ita_Latn'), ('nl', 'nld_Latn'),
    ('pt', 'por_Latn'), ('sv', 'swe_Latn'), ('bg', 'bul_Cyrl'), ('pl', 'pol_Latn'), ('en', 'eng_Latn')
]

def format_example(sample):
    question = sample["question"]
    paragraph = sample["flores_passage"]

    prompt = 'Read the text and answer the question based on it.\n'
    prompt += paragraph
    prompt += "\n"
    prompt += question
    prompt += "\nAnswer: "

    # for i in range(1, 5):
    #     prompt += f"\n{choices[i - 1]}. {sample[f'mc_answer{i}']}"
    
    # prompt += "\nFinish your answer with \"the answer is (X)\" where X is the correct letter choice.\n"
    
    return prompt

for lang_code, hf_lang in langs:
    prompts = []
    ground_truths = []
    predicted_answers = []
    model_outputs = []

    dataset = load_dataset('facebook/belebele', hf_lang, trust_remote_code=True)
    small_dataset = dataset['test'].select(range(30))

    for sample in tqdm(small_dataset, desc=f"Processing {lang_code}"):
        prompt = format_example(sample)
        conversation = [{"role": "user", "content": prompt}]
        full_prompt = tokenizer.apply_chat_template(
                                        conversation,
                                        tokenize=False,                # Don't tokenize
                                        add_generation_prompt=False    # Don't add assistant prefix unless needed
                                    )
        

        prompts.append(full_prompt)

        
        correct_answer = sample[f'mc_answer{sample["correct_answer_num"]}']
        ground_truths.append(correct_answer)
        
    prompts = [instr.replace("<|end_of_sequence|>", "") for instr in prompts]
    outputs = llm.generate(prompts, sampling_params)
    output_texts = [output.outputs[0].text for output in outputs]

    # for text in output_texts:
    #     pred = extract_answer(text, level='l2')
    #     if pred not in choices:
    #         pred = "N/A"
    #     predicted_answers.append(pred)
    #     model_outputs.append(text)

    # Evaluate
    # correct = sum([1 for gt, pred in zip(ground_truths, predicted_answers) if gt == pred])
    # total = len(ground_truths)
    # acc = correct / total

    # with open(f'{output_dir}/belebele_{lang_code}.txt', 'w') as fw:
    #     fw.write(str(acc))

    df = pd.DataFrame({
        'prompts': prompts,
        'answers': ground_truths,
        # 'predictions': predicted_answers,
        'model_output': output_texts
    })
    df.to_csv(f'{output_dir}/belebele_{lang_code}.tsv', sep='\t', index=False)
