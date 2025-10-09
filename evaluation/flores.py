import pandas as pd
from vllm import LLM, SamplingParams
from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
import argparse
import os



parser = argparse.ArgumentParser(description="Load model and language setup.")
parser.add_argument('--model_name', type=str, required=True, help='Path or identifier of the model to load.')
parser.add_argument('--model', type=str, required=True, help='Model alias or custom identifier.')

args = parser.parse_args()
output_dir = f"/scratch/cs/small_lm/eval_scripts/flores/{args.model}"
os.makedirs(output_dir, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

langs = [
    ('Czech', 'cs', 'ces_Latn'),
    ('German', 'de', 'deu_Latn'),
    ('Greek', 'el', 'ell_Grek'),
    ('Spanish', 'es', 'spa_Latn'),
    ('Finnish', 'fi', 'fin_Latn'),
    ('French', 'fr', 'fra_Latn'),
    ('Italian', 'it', 'ita_Latn'),
    ('Dutch', 'nl', 'nld_Latn'),
    ('Portuguese', 'pt', 'por_Latn'),
    ('Swedish', 'sv', 'swe_Latn'),
    ('Bulgarian', 'bg', 'bul_Cyrl'),
    ('Polish', 'pl', 'pol_Latn')
]

# Load Mistral-7B using vLLM
#model_name =  "HuggingFaceTB/SmolLM2-1.7B-Instruct" # "/scratch/cs/small_lm/gemma-3-1b-it" # "utter-project/EuroLLM-1.7B-Instruct" # "HuggingFaceTB/SmolLM2-1.7B-Instruct" # 
llm = LLM(model=args.model_name, dtype="float16", gpu_memory_utilization=.7,)
# model = 'ellm'


for lang in langs:

    data = load_dataset('Muennighoff/flores200', f"eng_Latn-{lang[2]}", trust_remote_code=True)
    
    data = data['dev'].to_pandas()

    # Translation prompt template
    PROMPT_TEMPLATE = f"Translate the following text from English to {lang[0]}: "
    SAMPLING_PARAMS = SamplingParams(temperature=0, max_tokens=200)

    print("Preparing prompts...")
    prompts = [PROMPT_TEMPLATE + sentence for sentence in tqdm(data[f"sentence_eng_Latn"].values)]
    prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
    full_prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False) for prompt in prompts]
    full_prompts = [instr.replace("<|end_of_sequence|>", "") for instr in full_prompts]

    # Generate translations in a batch
    print("Translating sentences...")
    outputs = llm.generate(full_prompts, SAMPLING_PARAMS)

    # Extract translations and add them to the data
    data["generated_translation"] = [output.outputs[0].text.strip().replace('\n', ' ') for output in outputs]

    # Load COMET model
    print("Loading COMET model...")
    # model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint('/scratch/cs/small_lm/eval_scripts/Unbabel--wmt22-comet-da/model.ckpt')

    # Function to compute COMET score
    def compute_comet_scores(src_sentences, mt_sentences, ref_sentences):
        data = [{"src": s, "mt": mt, "ref": ref} for s, mt, ref in zip(src_sentences, mt_sentences, ref_sentences)]
        scores = comet_model.predict(data, batch_size=8)
        return scores["scores"]  # List of individual sentence scores

    # Compute COMET scores for each translation
    print("Evaluating translations...")
    data["comet_score"] = compute_comet_scores(
        data[f"sentence_eng_Latn"].tolist(),
        data["generated_translation"].tolist(),
        data[f"sentence_{lang[2]}"].tolist(),
    )
    data.drop(columns=['URL', 'domain', 'has_image', 'has_hyperlink'], inplace=True)

    # Save results
    data.to_csv(f"/scratch/cs/small_lm/eval_scripts/flores/{args.model}/flores_{lang[1]}.tsv", index=False, sep='\t', quoting=3, escapechar='\\')


    # Print average COMET score
    avg_comet_score = data["comet_score"].mean()

    with open(f'/scratch/cs/small_lm/eval_scripts/flores/{args.model}/flores_{lang[1]}.txt', 'w') as fw:
        fw.write(str(avg_comet_score))
