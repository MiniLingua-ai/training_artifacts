import os
import pandas as pd
import psutil
import argparse
import logging
import re
from collections import defaultdict
from tqdm.auto import tqdm
from multiprocessing import Pool
from collections import defaultdict
import regex
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


for lang, val in language_distribution.items():
    language_distribution[lang] = val * language_mean[lang]

sm = sum(language_distribution.values()) / 0.95
for key, val in language_distribution.items():
    language_distribution[key] = val / sm

# # balanced
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


# # proposed
language_distribution = {
    'en': 0.3,
    'fr': 0.12,
    'pl': 0.04, 
    'de': 0.13,
    'nl': 0.03, 
    'sv': 0.01, 
    'pt': 0.05, 
    'es': 0.14, 
    'it': 0.06, 
    'fi': 0.01, 
    'el': 0.02, 
    'bg': 0.01, 
    'cs': 0.02
}

# # train
language_distribution = {
    'en': 0.25,
    'fr': 0.12,
    'pl': 0.04, 
    'de': 0.17,
    'nl': 0.04, 
    'sv': 0.01, 
    'pt': 0.07, 
    'es': 0.13, 
    'it': 0.06, 
    'fi': 0.01, 
    'el': 0.01, 
    'bg': 0.01, 
    'cs': 0.02
}


# # original fineweb
# language_distribution = {
#     'en': 0.45, # 5% code
#     'fr': 0.08,
#     'pl': 0.03, 
#     'de': 0.12,
#     'nl': 0.03, 
#     'sv': 0.01, 
#     'pt': 0.04, 
#     'es': 0.10, 
#     'it': 0.04, 
#     'fi': 0.01, 
#     'el': 0.01, 
#     'bg': 0.01, 
#     'cs': 0.02
# }


language_distribution["code"] = 0.02

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

tokeniser_regex = {"eng": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
                   "deu": r"""'(?i:s)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
                   "ita": r"""([ ]?(?i:\b(l|gl|d|s|c|m|t|n|v|dall|dell|nell|sull|un|bell|quell|sant|quest|quattr|tutt|senz|nient|mezz|com))'|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)""",
                   "fra": r"""([ ]?(?i:\b(c|j|l|m|n|s|t|d|qu)')|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)""",
                   "rest": r"""[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

}


def generate_language_count(total_gb=50, language_percentage=language_distribution):
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
    doc_path = df.iloc[0].metadata["file_path"]

    for row in df.id.values:
        doc_id = int(row.strip().rsplit('/', 1)[1])
        file_path_to_id[doc_path].add(doc_id)

    
    return file_path_to_id

def generator(folder="./../train_hq", code_folder="./../stack", batch_size=1_000_000):
    word_freq = defaultdict(int)
    text_count = 0
    dump_count = 0
    lang_to_n_chars = generate_language_count()
    
    for root, _, files in os.walk(folder):
        if files:
            paths = [os.path.join(root, file_) for file_ in files]
            logging.info(f"Found {len(files)} files in {root}")
            # process all files
            for path_ in paths:
                file_path_to_id = run_processing(path_)
            # with Pool(min(len(paths), 16)) as p:
            #     for data in p.imap_unordered(run_processing, paths):
            #         for key, vals in data.items():
            #             global_file_path_to_id[key].update(vals)
    
    
    # go over each file and tokenise
                for file_path, indices in file_path_to_id.items():
                    file_name = file_path.rsplit('/', 1)[-1]
                    for short_lang, lang in list(lang_mapping.items()):
                        if f"-{lang}" in file_name:
                            break
                    if lang not in file_path:
                        print(files)
                        raise Exception(f"No lang in filepath {file_path}")

                    if lang_to_n_chars[short_lang] <= 0:
                        logging.info(f"Skip file: {file_path}")
                        continue
                    logging.info(f"Reading file: {file_path}")
                    df = pd.read_parquet(file_path, columns=["text"])
                    indices = list(indices)
                    # if remaining_docs < len(indices):
                    #     indices = random.sample(indices, remaining_docs, seed=42)
                    file_texts = df.iloc[indices]["text"].values
                    for text in file_texts:
                        text = text.strip()
                        if not text:
                            continue
                        if lang_to_n_chars[short_lang] <= 0:
                            break
                        lang_to_n_chars[short_lang] -= len(text)
                        pattern = tokeniser_regex.get(lang, tokeniser_regex["rest"])
                        if lang in {"ita", "fra"}:
                            words = list(zip(*regex.findall(pattern, text)))[0]
                        else:
                            words = regex.findall(pattern, text)


                        for word in words:
                            word_freq[word] += 1
                        text_count += 1
                        
                        if text_count >= batch_size:
                            dump_count += 1
                            dump_to_tsv(word_freq, dump_count)
                            word_freq.clear()
                            text_count = 0

                    logging.info(f"Processed {text_count} files")

    for root, _, files in os.walk(code_folder):
        for code in code_distibution:
            if code in root:
                break
        
        if code not in root:
            continue
        
        for file in files:
            if not file.endswith(".parquet"):
                continue
            file_path = os.path.join(root, file)
            if lang_to_n_chars[code] <= 0:
                logging.info(f"Skip file: {file_path}")
                continue

            logging.info(f"Reading file: {file_path}")
            df = pd.read_parquet(file_path, columns=["text"])
            file_texts = df["text"].values
            for text in file_texts:
                text = text.strip()
                if not text:
                    continue
                if lang_to_n_chars[code] <= 0:
                    break
                lang_to_n_chars[code] -= len(text)
                pattern = tokeniser_regex["rest"]
                words = regex.findall(pattern, text)

                for word in words:
                    word_freq[word] += 1
                text_count += 1
                        
                if text_count >= batch_size:
                    dump_count += 1
                    dump_to_tsv(word_freq, dump_count)
                    word_freq.clear()
                    text_count = 0

                logging.info(f"Processed {text_count} files")


    if word_freq:
        dump_count += 1
        dump_to_tsv(word_freq, dump_count)


pattern = r'^[^\t\n]+\t\d+\n$'

def dump_to_tsv(word_freq, dump_count):
    output_file = f"./hq_train/word_frequencies_{dump_count}.tsv"
    with open(output_file, "w", encoding="utf-8") as f:
        for word, count in word_freq.items():
            write_str = f"{word}\t{count}\n"
            if bool(re.match(pattern, write_str)):
                f.write(write_str)
    logging.info(f"Dumped {len(word_freq)} word frequencies to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="./../train_web", help="Folder containing Parquet files")
    parser.add_argument("--batch_size", type=int, default=1_000_000, help="Batch size before dumping word frequencies")
    args = parser.parse_args()
    
    generator(folder=args.folder, batch_size=args.batch_size)
