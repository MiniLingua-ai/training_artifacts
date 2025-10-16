import os
import pandas as pd

# Define the parent folder containing the subfolders
parent_folder = "/scratch/cs/small_lm/wikipedia"
langs = ['en', 'cs',  'de', 'pl', 'fi', 'es',  'fr', 'sv']

print('process started')
# Iterate over all subfolders in the parent folder
for child_folder in langs:
    child_path = os.path.join(parent_folder, child_folder)
    print(child_path)
    
    # Skip if it's not a directory
    if not os.path.isdir(child_path):
        continue

    print(f"Processing folder: {child_folder}")

    # Initialize a counter for parquet files in the current folder
    file_counter = 0

    # Iterate over all parquet files in the subfolder
    for parquet_file in os.listdir(child_path):
        
        parquet_path = os.path.join(child_path, parquet_file)

        # Read the parquet file
        df = pd.read_parquet(parquet_path)

        # Select only the 'text' column
        if 'text' not in df.columns:
            print(f"Warning: 'text' column not found in {parquet_path}. Skipping.")
            continue
        
        text_data = df[['text']]

        # Create output file name
        output_file = os.path.join(parent_folder, f"{child_folder}/{child_folder}_{file_counter}.parquet")
        
        # Save the new parquet file
        text_data.to_parquet(output_file, index=False)
        print(f"Saved {output_file}")

        file_counter += 1