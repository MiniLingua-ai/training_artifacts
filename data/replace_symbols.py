import pandas as pd
import os
import logging
from multiprocessing import Pool

exclude_folders = ["bad_words_data", "datatrove_output", "datatrove_output_boris", 
                   "datatrove_scripts", "datatrove_scripts_new", "miniconda3", "myenv", 
                   "test_datasets", "test_web", "to_check", "tokeniser", "tokenizers", "train_web", "web_data_scripts", "stack"]

all_file_pathes = set()

for root, _, files in os.walk("."):
        if any([folder in root for folder in exclude_folders]):
            continue
        for file in files:
            if file.endswith(".parquet"):
                file_path = os.path.join(root, file)
                all_file_pathes.add(file_path)

print(len(all_file_pathes))


def process_parquet(file_path):
    try:
        print(file_path)
        # Read the Parquet file
        df = pd.read_parquet(file_path).dropna()

        # Check if 'text' column exists
        if 'text' in df.columns:
            df['text'] = df['text'].apply(lambda x: x.replace('`', "'").replace('’', "'"))\
            # Write the modified data back to Parquet
            df.to_parquet(file_path)
            logging.info(f"Processed: {file_path}")
        else:
            logging.error(f"No text field {file_path}: {e}")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")


with Pool(128) as p:
    res = p.map(process_parquet, all_file_pathes)