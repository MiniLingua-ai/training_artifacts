import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import argparse
import os




parser = argparse.ArgumentParser(description="Load model and language setup.")
parser.add_argument('--model_name', type=str, required=True, help='Path or identifier of the model to load.')
parser.add_argument('--model', type=str, required=True, help='Model alias or custom identifier.')

args = parser.parse_args()

output_dir = f"/scratch/cs/small_lm/eval_scripts/mmlu/{args.model}"
os.makedirs(output_dir, exist_ok=True)

#model_name =  "/scratch/cs/small_lm/gemma-3-1b-it" #"HuggingFaceTB/SmolLM2-1.7B-Instruct" #"utter-project/EuroLLM-1.7B-Instruct"
llm = LLM(model=args.model_name, dtype="float16")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# model = 'gemma'

# Define multiple-choice options
choices = ["A", "B", "C", "D"] 
mid_letters = np.array(["  A", "  B", "  C", "  D"])
choice_tokens = [tokenizer.convert_tokens_to_ids(token) for token in choices] \
    + [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token)[1]) for token in mid_letters] 

choices = np.array(choices + choices)

configs = ['abstract_algebra_BG', 'anatomy_BG', 'astronomy_BG',
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


langs =  ['CS', 'DE', 'EL', 'ES', 'FI', 'FR', 'IT', 'NL', 'SV', 'BG', 'PL', 'PT-PT']

def softmax(logits):
    """Compute softmax probabilities from log probabilities."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def format_example(sample, include_answer=True):
    """Format a single MMLU example into a structured prompt."""
    prompt = sample["question"]
    for i, option in enumerate(sample["choices"]):
        prompt += f"\n{choices[i]}. {option}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {choices[sample['answer']]}\n\n"
    return prompt

def gen_prompt(train_samples, subject, k=3):
    """Generate a few-shot prompt using `k` examples from training data."""
    prompt = f"The following are multiple-choice questions (with answers) about {' '.join(subject.split('/')[0].split('_'))}.\n\n"
    for i in range(min(k, len(train_samples))):
        prompt += format_example(train_samples[i], include_answer=True)
    return prompt


for lang in langs:
    prompts = []
    ground_truths = []  # Store correct answers for evaluation
    for config in configs:
        config = config.replace('BG', lang)

        dataset = load_dataset('openGPT-X/mmlux', config, trust_remote_code=True)

        for sample in tqdm(dataset['test'], desc="Generating Prompts"):
            subject = sample['id']

            # Select 3 training examples from the same subject
            subject_train_data = dataset['dev'].filter(lambda x: x['id'] == subject)
            few_shot_examples = subject_train_data.shuffle(seed=42).select(range(min(3, len(subject_train_data))))

            train_prompt = gen_prompt(few_shot_examples, subject)
            test_prompt = format_example(sample, include_answer=False)
            full_prompt = train_prompt + test_prompt

            prompts.append(full_prompt)
            ground_truths.append(choices[sample["answer"]])  # Store correct answer




    sampling_params = SamplingParams(temperature=0, max_tokens=1, logprobs=4, top_k=4)


    # Run batch inference
    outputs = llm.generate(prompts, sampling_params)

    logprobs_dicts = [output.outputs[0].logprobs[0] for output in outputs]

    # Convert logprobs to a matrix: shape (batch_size, 4)

    batch_log_probs = np.array([
        [logprobs_dict[choice].logprob if choice in logprobs_dict else -100 for choice in choice_tokens]
        for logprobs_dict in logprobs_dicts
    ])

    # Apply softmax row-wise
    batch_probs = np.exp(batch_log_probs - np.max(batch_log_probs, axis=1, keepdims=True))
    batch_probs /= np.sum(batch_probs, axis=1, keepdims=True)

    # Get predicted answers
    batch_predicted_indices = np.argmax(batch_probs, axis=1).astype(int)  # Ensure integer type
    batch_predicted_answers = choices[batch_predicted_indices]

    acc = accuracy_score(ground_truths, batch_predicted_answers)


    with open(f'/scratch/cs/small_lm/eval_scripts/mmlu/{args.model}/mmlu_{lang.lower()}.txt', 'w') as fw:
        fw.write(str(acc))


    df = pd.DataFrame({'prompts': prompts,
                    'answers': ground_truths,
                    'predictions': batch_predicted_answers})

    df.to_csv(f'/scratch/cs/small_lm/eval_scripts/mmlu/{args.model}/mmlu_{lang.lower()}.tsv', sep='\t', index=False)