import sentencepiece as spm
import os
import pandas as pd
import logging
from collections import defaultdict
from multiprocessing import Pool
from collections import defaultdict
import argparse
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='Process text data with a SentencePiece tokenizer.')
parser.add_argument('--tokenizer', type=str, required=True, help='Name of the SentencePiece tokenizer model')
args = parser.parse_args()

# Load SentencePiece model here (or any other tokenizer)
sp_model = spm.SentencePieceProcessor()
sp_model.load(f"/scratch/cs/small_lm/tokeniser_training/{args.tokenizer}.model")

lang_mapping = {
    'bg': 'bul',  # Bulgarian
    'cs': 'ces',  # Czech
    'nl': 'nld',  # Dutch
    'de': 'deu',  # German
    'el': 'ell',  # Greek
    'it': 'ita',  # Italian
    'pl': 'pol',  # Polish
    'fi': 'fin',  # Finnish
    'es': 'spa',  # Spanish
    'pt': 'por',  # Portuguese
    'fr': 'fra',  # French
    'sv': 'swe',  # Swedish
    'en': 'eng'   # English
}

language_distribution = {
    'fr': 212613103,
    'en': 268884897, 
    'pl': 78216990, 
    'de': 260580217,
    'nl': 85665508, 
    'sv': 28459326, 
    'pt': 137742328, 
    'es': 244732239, 
    'it': 138888100, 
    'fi': 20376289, 
    'el': 21177763, 
    'bg': 14549461, 
    'cs': 37652931
}

language_mean = {
    "en": 524,
    "pl": 400,
    "nl": 421,
    "fr": 540,
    "pt": 470,
    "de": 490,
    "sv": 621,
    "it": 516,
    "fi": 417,
    "es": 535,
    "el": 408,
    "bg": 566,
    "cs": 480
}


language_distribution = {
    'fr': 0.12,
    'en': 0.2, 
    'pl': 0.04, 
    'de': 0.13,
    'nl': 0.04, 
    'sv': 0.04, 
    'pt': 0.05, 
    'es': 0.14, 
    'it': 0.06, 
    'fi': 0.03, 
    'el': 0.03, 
    'bg': 0.03, 
    'cs': 0.03
}

language_distribution["code"] = 0.05

code_distibution = {
    "Markdown": 0.15,
    "SQL": 0.1,
    "XML": 0.1,
    "JSON": 0.25,
    "YAML": 0.1,
    "TeX": 0.08,
    "C": 0.1,
    "Python": 0.1,
    "Shell": 0.02
}


def generate_language_count(total_gb=2, language_percentage=language_distribution):
    # gb to bytes (I assume 1 char = 1 byte) bg, el - 2 but i count them as 1
    total_docs = total_gb * 1024 * 1024 * 1024  #/ (250 * 8)
    lang_to_n_chars = {}
    for lang, fraction in language_percentage.items():
        if lang == "code":
            for code, code_fraction in code_distibution.items():
                lang_to_n_chars[code] = round(fraction * total_docs * code_fraction)
        else:
            lang_to_n_chars[lang] = round(fraction * total_docs)
    return lang_to_n_chars


def run_processing(file):
    logging.info(f"Processing file: {file}")
    file_path_to_id = defaultdict(set)
    df = pd.read_parquet(file, columns=["id", "metadata"])
    base = df.iloc[0]['metadata']
    if isinstance(base, str):
        base = json.loads(base)
        
    doc_path = base["file_path"]

    for row in df.id.values:
        doc_id = int(row.strip().rsplit('/', 1)[1])
        file_path_to_id[doc_path].add(doc_id)

    
    return file_path_to_id

def code_generator(code_folder="/scratch/cs/small_lm/test_code"):
    lang_to_n_chars = generate_language_count()
    
    for root, _, files in os.walk(code_folder):
        if files:
            paths = [os.path.join(root, file_) for file_ in files]
            logging.info(f"Found {len(files)} files in {root}")
            # process all files
            for path_ in paths:
                file_path_to_id = run_processing(path_)

    
                # go over each file and tokenise
                for file_path, indices in file_path_to_id.items():
                    file_name = file_path.rsplit('/')[-2]
                    for lang in list(code_distibution.keys()):
                        if f"{lang}" in file_name:
                            break
                        
                    
                    if lang_to_n_chars[lang] <= 0:
                        logging.info(f"Skip file: {file_path}")
                        continue
                    logging.info(f"Reading file: {file_path}")

                    if not os.path.exists(file_path):
                        print(f"Skipping {file_path} (file not found)")
                        continue

                    df = pd.read_parquet(file_path, columns=["text"])
                    indices = list(indices)

                    file_texts = df.iloc[indices]["text"].values
                    for text in file_texts:
                        text = text.strip()
                        if not text:
                            continue
                        if lang_to_n_chars[lang] <= 0:
                            break

                        lang_to_n_chars[lang] -= len(text)

                        yield lang, text



def generator(folder="./../test_web"):
    lang_to_n_chars = generate_language_count()
    
    for root, _, files in os.walk(folder):
        if files:
            paths = [os.path.join(root, file_) for file_ in files]
            logging.info(f"Found {len(files)} files in {root}")
            # process all files
            for path_ in paths:
                file_path_to_id = run_processing(path_)

    
                # go over each file and tokenise
                for file_path, indices in file_path_to_id.items():
                    file_name = file_path.rsplit('/', 1)[-1]
                    for short_lang, lang in list(lang_mapping.items()):
                        if f"-{lang}" in file_name:
                            break
                    
                    if lang_to_n_chars[short_lang] <= 0:
                        logging.info(f"Skip file: {file_path}")
                        continue
                    logging.info(f"Reading file: {file_path}")
                    df = pd.read_parquet(file_path, columns=["text"])
                    indices = list(indices)

                    file_texts = df.iloc[indices]["text"].values
                    for text in file_texts:
                        text = text.strip()
                        if not text:
                            continue
                        if lang_to_n_chars[short_lang] <= 0:
                            break

                        lang_to_n_chars[short_lang] -= len(text)

                        yield lang, text

# create dict langs wordcount
langs_wordcount = defaultdict(int)

for lang, text in generator():
    count = len(sp_model.encode(text, out_type=str))
    langs_wordcount[lang] += count

for lang, text in code_generator("./../test_code"):
    count = len(sp_model.encode(text, out_type=str))
    langs_wordcount[lang] += count

df = pd.DataFrame.from_dict(langs_wordcount, orient='index', columns=['word_count'])
df.index.name = 'language'
df.to_csv(f'{args.tokenizer}_wordcount.tsv', sep='\t')