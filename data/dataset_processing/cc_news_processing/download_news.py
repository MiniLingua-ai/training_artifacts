import os
import pandas as pd
from datasets import load_dataset


languages = ['de'] # Set list of languages here
output_folder = "/scratch/cs/small_lm" # Set base output folder here
lines_per_file = 50000 # Set number of lines per file here

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load and process the dataset for each language
for lang in languages:
    print(f"Processing language: {lang}")

    # Load the dataset for the current language
    dataset = load_dataset("intfloat/multilingual_cc_news", languages=[lang], split="train", cache_dir=output_folder, data_dir=output_folder, trust_remote_code=True)
    
    # Check if the column 'maintext' exists
    if 'maintext' not in dataset.column_names:
        print(f"Warning: 'maintext' column not found for language {lang}. Skipping.")
        continue

    # Extract the 'maintext' column and rename it to 'text'
    texts = dataset['maintext']

    # Initialize variables
    buffer = []
    file_counter = 0

    # Process the dataset in chunks
    for text in texts:
        buffer.append(text)

        # When buffer reaches the specified size, write to a Parquet file
        if len(buffer) == lines_per_file:
            output_file = os.path.join(output_folder, f"cc_news/{lang}_{file_counter}.parquet")
            pd.DataFrame({"text": buffer}).to_parquet(output_file, index=False)
            print(f"Saved {lines_per_file} lines to {output_file}")
            buffer = []  # Reset the buffer
            file_counter += 1

    # Save any remaining lines in the buffer
    if buffer:
        output_file = os.path.join(output_folder, f"cc_news/{lang}_{file_counter}.parquet")
        pd.DataFrame({"text": buffer}).to_parquet(output_file, index=False)
        print(f"Saved {len(buffer)} remaining lines to {output_file}")


