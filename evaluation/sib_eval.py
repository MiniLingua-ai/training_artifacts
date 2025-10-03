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

output_dir = f"/scratch/cs/small_lm/eval_scripts/sib/{args.model}"
os.makedirs(output_dir, exist_ok=True)

# Model and tokenizer setup
#model_name = "/scratch/cs/small_lm/gemma-3-1b-it"  # "HuggingFaceTB/SmolLM2-1.7B-Instruct" #"utter-project/EuroLLM-1.7B-Instruct"
llm = LLM(model=args.model_name, dtype="float16")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
sampling_params = SamplingParams(temperature=0, max_tokens=100, logprobs=10, top_k=4)
langs = [('cs', 'ces_Latn'), ('de', 'deu_Latn'), ('el', 'ell_Grek'), ('es', 'spa_Latn'),
 ('fi', 'fin_Latn'), ('fr', 'fra_Latn'), ('it', 'ita_Latn'), ('nl', 'nld_Latn'),
 ('pt', 'por_Latn'), ('sv', 'swe_Latn'), ('bg', 'bul_Cyrl'), ('pl', 'pol_Latn'), ('en', 'eng_Latn')]
# model = 'gemma'


# Define categories and their letter mappings
categories = [
    "science/technology", "travel", "politics", "sports", "health", 
    "entertainment", "geography"
]
category_to_letter = {category: chr(65 + i) for i, category in enumerate(categories)}  # Map to A, B, C, D, etc.
#letters = list(category_to_letter.values())  # ["A", "B", "C", "D", ...]
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G'] + ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# choices = ["A", "B", "C", "D", "E", "F", "G"] 
# mid_letters = np.array(["  A", "  B", "  C", "  D",  "  E", "  F", "  G"])
# category_tokens = [tokenizer.convert_tokens_to_ids(token) for token in choices] \
#     + [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token)[1]) for token in mid_letters] 




choices = ["A", "B", "C", "D", "E", "F", "G"] 
mid_letters = ["  A", "  B", "  C", "  D", "  E", "  F", "  G"]
choice_tokens = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))[0] for token in choices] \
    + [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token)[1]) for token in mid_letters] 

choices = np.array(choices + choices)



# Function to compute softmax
def softmax(logits):
    """Compute softmax probabilities from log probabilities."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

# Function to format each example into a prompt
def format_example(sample):
    """Format the text to ask the model to choose the correct topic."""
    prompt = "Given the text below, choose the most appropriate topic from the following options:\n\n"
    for idx, category in enumerate(categories):
        prompt += f"{letters[idx]}. {category}\n"
    prompt += "\nText:\n"
    prompt += sample["text"]
    prompt += "\nAnswer:\n"
    return prompt


for lang in langs:
    # Initialize lists for prompts and ground truth answers
    prompts = []
    ground_truths = []  # Store correct answers for evaluation

    # Load the dataset
    dataset = load_dataset('Davlan/sib200', lang[1])

    # Loop over the dataset to generate prompts and store ground truth answers
    for sample in tqdm(dataset['test'], desc="Generating Prompts"):

        full_prompt = format_example(sample)

        prompts.append(full_prompt)
        # Map the category to the corresponding letter (A, B, C, etc.)
        ground_truths.append(category_to_letter.get(sample["category"]))  # Default to 'A' if not found

    # Run batch inference
    outputs = llm.generate(prompts, sampling_params)

    logprobs_dicts = [output.outputs[0].logprobs[0] for output in outputs]
    answer_predicts = []
    for dic in logprobs_dicts:
        flag = 0
        for item in dic.values():
            if item.decoded_token.strip() in ["A", "B", "C", "D", "E", "F", "G"]:
                answer_predicts.append(item.decoded_token.strip())
                flag = 1
                break
        if flag:
            continue
        else:
            answer_predicts.append('N/A')
    texts = [output.outputs[0].text for output in outputs]
    # # Convert logprobs to a matrix: shape (batch_size, num_categories)
    # # print(logprobs_dicts[10])
    # batch_log_probs = np.array([
    #         [logprobs_dict[choice].logprob if choice in logprobs_dict else -100 for choice in choice_tokens]
    #         for logprobs_dict in logprobs_dicts
    #     ])
        
    # Apply softmax row-wise
    # batch_probs = np.exp(batch_log_probs - np.max(batch_log_probs, axis=1, keepdims=True))
    # batch_probs /= np.sum(batch_probs, axis=1, keepdims=True)

    # # Get predicted answers
    # batch_predicted_indices = np.argmax(batch_probs, axis=1).astype(int)  # Ensure integer type
    # batch_predicted_answers = np.array(letters)[batch_predicted_indices]


    # Calculate accuracy
    acc = accuracy_score(ground_truths, answer_predicts)

    # Save the accuracy to a text file
    with open(f'/scratch/cs/small_lm/eval_scripts/sib/{args.model}/sib_{lang[0]}.txt', 'w') as fw:
        fw.write(f"{acc}")

    # Save the results in a DataFrame and write to a CSV file
    df = pd.DataFrame({'prompts': prompts,
                    'answers': ground_truths,
                    'predictions': answer_predicts,
                    'logprobs': logprobs_dicts,
                    'texts': texts})

    df.to_csv(f'/scratch/cs/small_lm/eval_scripts/sib/{args.model}/sib_{lang[0]}.tsv', index=False, sep='\t')
    # break

