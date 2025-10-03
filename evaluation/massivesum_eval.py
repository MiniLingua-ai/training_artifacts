import pandas as pd
from vllm import LLM, SamplingParams
import evaluate
from tqdm import tqdm
import argparse
import os



parser = argparse.ArgumentParser(description="Load model and language setup.")
parser.add_argument('--model_name', type=str, required=True, help='Path or identifier of the model to load.')
parser.add_argument('--model', type=str, required=True, help='Model alias or custom identifier.')

args = parser.parse_args()
output_dir = f"/scratch/cs/small_lm/eval_scripts/massivesum/{args.model}"
os.makedirs(output_dir, exist_ok=True)

langs = [
    ('Czech', 'cs', 'ces_Latn'),
    ('German', 'de', 'deu_Latn'),
    ('Greek', 'el', 'ell_Grek'),
    ('Spanish', 'es', 'spa_Latn'),
    # ('Finnish', 'fi', 'fin_Latn'),
    ('French', 'fr', 'fra_Latn'),
    ('Italian', 'it', 'ita_Latn'),
    ('Dutch', 'nl', 'nld_Latn'),
    ('Portuguese', 'pt', 'por_Latn'),
    ('Swedish', 'sv', 'swe_Latn'),
    ('Bulgarian', 'bg', 'bul_Cyrl'),
    ('Polish', 'pl', 'pol_Latn'),
    # ('English', 'en', 'eng_Latn')
]

# Load Mistral-7B using vLLM
#model_name =  "HuggingFaceTB/SmolLM2-1.7B-Instruct" # "/scratch/cs/small_lm/gemma-3-1b-it" # "utter-project/EuroLLM-1.7B-Instruct" # "HuggingFaceTB/SmolLM2-1.7B-Instruct" # 
llm = LLM(model=args.model_name, dtype="float16", gpu_memory_utilization=.7,)
# model = 'ellm'
rouge = evaluate.load('rouge')
data_full = pd.read_parquet('/scratch/cs/small_lm/eval_scripts/massivesum.parquet')


for lang in langs:

    data = data_full[data_full['language'] == lang[2].split('_')[0]].copy()
    data['words'] = data['text'].apply(lambda x: len(x.split()))
    data = data[data['words'] < 800].head(100).copy()

    # Translation prompt template
    PROMPT_TEMPLATE = f"Summarize the following text:\n"
    SAMPLING_PARAMS = SamplingParams(temperature=0, max_tokens=2048)

    print("Preparing prompts...")
    prompts = [PROMPT_TEMPLATE + sentence for sentence in tqdm(data["text"].values)]

    # Generate translations in a batch
    print("Summarizing sentences...")
    outputs = llm.generate(prompts, SAMPLING_PARAMS)

    # Extract translations and add them to the data
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
    data.to_csv(f"/scratch/cs/small_lm/eval_scripts/massivesum/{args.model}/massivesum_{lang[1]}.tsv", index=False, sep='\t')


    # Print average COMET score
    avg_comet_score = data["rouge2"].mean()

    with open(f'/scratch/cs/small_lm/eval_scripts/massivesum/{args.model}/massivesum_{lang[1]}.txt', 'w') as fw:
        fw.write(str(avg_comet_score))
