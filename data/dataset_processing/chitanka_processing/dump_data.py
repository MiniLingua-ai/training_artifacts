import os
import pandas as pd

# Define the path to the 'chitanka' folder
base_path = "chitanka"

# Initialize variables
texts = []  # To store texts
file_counter = 1  # To track the file index for parquet files
token_counter = 0

# Walk through all subfolders and files
for root, _, files in os.walk(base_path):
    for file in files:
        if file.endswith(".txt"):
            # Read the content of the text file
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                texts.append(text)
                token_counter += len(text.split())

            # If we've accumulated 5000 texts, write them to a parquet file
            if len(texts) == 5000:
                output_file = f"bg_{file_counter}.parquet"
                pd.DataFrame({"text": texts}).to_parquet(output_file, index=False)
                print(f"Dumped {len(texts)} texts to {output_file}")
                texts = []  # Reset the list
                file_counter += 1

# Write any remaining texts to a final parquet file
if texts:
    output_file = f"bg_{file_counter}.parquet"
    pd.DataFrame({"text": texts}).to_parquet(output_file, index=False)
    print(f"Dumped {len(texts)} texts to {output_file}")

all_files = 5000 * file_counter + len(texts)
print(f"Calculated {token_counter} tokens in {all_files} files")
