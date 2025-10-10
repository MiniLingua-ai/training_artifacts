import pandas as pd
from vllm import LLM, SamplingParams
import evaluate
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import os



parser = argparse.ArgumentParser(description="Load model and language setup.")
parser.add_argument('--model_name', type=str, required=True, help='Path or identifier of the model to load.')
parser.add_argument('--model', type=str, required=True, help='Model alias or custom identifier.')
parser.add_argument('--base_path', type=str, required=True, help='Base directory for input/output paths.')

args = parser.parse_args()

# Create output directory
output_dir = os.path.join(args.base_path, "massivesum", args.model)
os.makedirs(output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

langs = [
    ('Czech', 'cs', 'ces_Latn'),
    ('German', 'de', 'deu_Latn'),
    ('Greek', 'el', 'ell_Grek'),
    ('Spanish', 'es', 'spa_Latn'),
    # ('Finnish', 'fi', 'fin_Latn'), Missing from MassiveSum
    ('French', 'fr', 'fra_Latn'),
    ('Italian', 'it', 'ita_Latn'),
    ('Dutch', 'nl', 'nld_Latn'),
    ('Portuguese', 'pt', 'por_Latn'),
    ('Swedish', 'sv', 'swe_Latn'),
    ('Bulgarian', 'bg', 'bul_Cyrl'),
    ('Polish', 'pl', 'pol_Latn')
]

# Load LLM using vLLM
llm = LLM(model=args.model_name, dtype="float16", gpu_memory_utilization=.7,)
rouge = evaluate.load('rouge')

# Original dataset is here: https://github.com/danielvarab/massive-summ.
data_path = os.path.join(args.base_path, "massivesum.parquet")
data_full = pd.read_parquet(data_path)

for lang in langs:

    data = data_full[data_full['language'] == lang[2].split('_')[0]].copy()
    data['words'] = data['text'].apply(lambda x: len(x.split()))
    # We used subsample fitting into 2048 tokens context window
    data = data[data['words'] < 500].head(100).copy()
    print(data.shape)

    # Translation prompt template
    PROMPT_TEMPLATE = f"Summarize the following text in a couple of sentences:\n"
    SAMPLING_PARAMS = SamplingParams(temperature=0, max_tokens=2048)

    print("Preparing prompts...")
    prompts = [PROMPT_TEMPLATE + sentence for sentence in tqdm(data["text"].values)]
    prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
    full_prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False) for prompt in prompts]
    full_prompts = [instr.replace("<|end_of_sequence|>", "") for instr in full_prompts]

    # Generate translations in a batch
    print("Summarizing sentences...")
    outputs = llm.generate(full_prompts, SAMPLING_PARAMS)

    # Extract translations and add them to the data
    data['prompts'] = full_prompts
    data['summary'] = [i.strip().replace('\n', ' ') for i in data['summary'].tolist()]
    data["generated_summary"] = [output.outputs[0].text.strip().replace('\n', ' ') for output in outputs]
    print(outputs[0].outputs[0].text)

    

    def compute_rouge_scores(predictions, references):
        results = rouge.compute(predictions=predictions,
                         references=references,
                         use_aggregator=False)
        return results['rouge2'][0]

    # Compute COMET scores for each translation
    print("Evaluating translations...")
    data["rouge2"] = compute_rouge_scores(
        data["generated_summary"].tolist(),
        data[f"summary"].tolist(),
    )
    
    # Save results
    data['text'] = data['text'].apply(lambda x: x.replace('\t', ' '))
    parquet_path = os.path.join(output_dir, f"massivesum_{lang[1]}.parquet")
    data.to_parquet(parquet_path, index=False)

    # Print and save average ROUGE score
    avg_rouge_score = data["rouge2"].mean()
    txt_path = os.path.join(output_dir, f"massivesum_{lang[1]}.txt")
    with open(txt_path, 'w') as fw:
        fw.write(str(avg_rouge_score))
