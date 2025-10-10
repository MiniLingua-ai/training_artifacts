import pandas as pd
from tqdm.auto import tqdm
import os

# Input folder and output details
input_folder = "/scratch/cs/small_lm/cs_syn_corpus/cs_syn_corpus"  # Replace with your input folder path
output_folder = "/scratch/cs/small_lm/cs_syn_corpus"  # Replace with your output folder
os.makedirs(output_folder, exist_ok=True)

# Parameters
files_per_parquet = 5000  # Number of text files per Parquet
file_count = 49320  # Total number of text files

# Variables
parquet_index = 0
current_batch = []
count = 0

# Function to write the batch to Parquet dynamically
def write_to_parquet(batch, index):
    if batch:
        df = pd.DataFrame(batch, columns=["text"])
        parquet_path = os.path.join(output_folder, f"cs_{index:03d}.parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"Written: {parquet_path}")

# Process each text file
for i in tqdm(range(file_count)):
    file_path = os.path.join(input_folder, f"{i}.txt")
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            # count += len(line.split())
            current_batch.append({"text": line.strip()})
    
    # Write to Parquet after processing a batch
    if (i + 1) % files_per_parquet == 0 or i == file_count - 1:
        write_to_parquet(current_batch, parquet_index)
        current_batch = []  # Clear the batch
        parquet_index += 1

# print(f"Processed {count} number of tokens")
