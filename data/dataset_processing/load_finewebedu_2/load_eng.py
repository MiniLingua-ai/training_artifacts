# Script for loading dataset for any language passed as command-line argument.
import os
import sys
import argparse
from datasets import load_dataset, DownloadConfig, Dataset
from huggingface_hub import login
from tqdm import tqdm
import logging
from settings import SETTINGS

def main():
    # Language code mapping to ISO codes
    LANGUAGE_MAPPING = {
        'bg': 'bg',
        'cs': 'cs',
        'nl': 'nl', 
        'en': 'en',
        'fi': 'fi',
        'fr': 'fr',
        'de': 'de',
        'el': 'el',
        'it': 'it',
        'pl': 'pl',
        'pt': 'pt',
        'es': 'es',
        'sv': 'sv'
    }
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Load and process dataset for a specific language')
    parser.add_argument('language', type=str, help='Language code (e.g., en, fr, de, etc.)')
    parser.add_argument('--batch-size', type=int, default=100000, help='Batch size for processing (default: 100000)')
    parser.add_argument('--file-size', type=int, default=500000, help='Number of examples per output file (default: 500000)')
    
    args = parser.parse_args()
    language_input = args.language.lower()
    batch_size = args.batch_size
    file_size = args.file_size
    
    # Validate and get ISO code
    if language_input not in LANGUAGE_MAPPING:
        print(f"Error: Unsupported language '{args.language}'")
        print("Supported ISO codes:", list(set([v for v in LANGUAGE_MAPPING.values()])))
        sys.exit(1)
    
    language = language_input  # Keep original for directory naming
    iso_code = LANGUAGE_MAPPING[language_input]
    
    os.environ['HF_HOME'] = '/scratch/cs/small_lm/cache/'

    config = DownloadConfig(
        cache_dir="/scratch/cs/small_lm/fine_web_edu_2",
        extract_on_the_fly=True,
        delete_extracted=True,
        num_proc=20
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=f'culturax_{language}.log',
        filemode='w'
    )

    login(SETTINGS["hf"]["token"])

    logging.info(f"Starting loading {language}")
    ds = load_dataset("HuggingFaceFW/fineweb",
                      name="sample-350BT",
                      split="train",
                      cache_dir="/scratch/cs/small_lm/fine_web_edu_2",
                      keep_in_memory=False,
                      download_config=config,
                      streaming=True)

    dir_name = f"/scratch/cs/small_lm/fine_web_edu_2/{language}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    batch = []
    full_dataset = []
    initial_number = 0
    file_name = f"web-edu-{language}-train-{{num}}.parquet"
    last_file = None

    for ex in tqdm(ds):
        batch.append({"text": ex["text"]})
        if len(batch) >= batch_size:
            full_dataset.extend(batch)
            batch = []
            if len(full_dataset) > file_size:
                logging.info(f"Processed file number {initial_number}")
                tmp_dataset = Dataset.from_list(full_dataset)
                tmp_dataset.to_parquet(os.path.join(dir_name, file_name.format(lang=iso_code, num=initial_number)))
                full_dataset = []
                initial_number += 1

    if len(batch) > 0:
        full_dataset.extend(batch)
        tmp_dataset = Dataset.from_list(full_dataset)
        tmp_dataset.to_parquet(os.path.join(dir_name, file_name.format(lang=iso_code, num=initial_number)))

    logging.info(f"{language} is loaded.")


if __name__ == "__main__":
    main()
        