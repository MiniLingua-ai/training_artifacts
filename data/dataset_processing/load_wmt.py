from datasets import load_dataset, Dataset
import os
import pandas as pd
from tqdm import tqdm

# # Load the WMT17 dataset using streaming (change if needed to "cs-en")
# dataset = load_dataset('wmt/wmt17', 'fi-en', split='train', streaming=True)

# # Create a directory to save the parquet files
# os.makedirs('wmt17', exist_ok=True)

# # Save the text column to parquet files

# save_data = []

# for example in tqdm(dataset):
#     save_data.append({"text" :example["translation"]['fi']})

# tmp_dataset = Dataset.from_list(save_data)
# tmp_dataset.to_parquet(os.path.join("wmt17", "fi.parquet"))


count = 0

dataset = load_dataset('wikimedia/wikipedia', '20231101.nl', split='train', streaming=True)


for example in tqdm(dataset):
    count += len(example["text"].split(' '))

print(count)
