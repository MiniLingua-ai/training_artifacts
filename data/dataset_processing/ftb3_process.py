import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# Folder containing text files
INPUT_FOLDER = "."  # Change this to your folder path
BATCH_SIZE = 30000  # Number of lines per Parquet file

def process_files(folder_path):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    batch = []
    file_idx = 0
    total_lines = 0

    for file in all_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                batch.append(line.strip())  # Remove any leading/trailing whitespace
                total_lines += 1

                if len(batch) >= BATCH_SIZE:
                    save_to_parquet(batch, file_idx)
                    file_idx += 1
                    batch = []

    # Save any remaining lines in the final batch
    if batch:
        save_to_parquet(batch, file_idx)

    print(f"Processing complete. Total lines processed: {total_lines}")

def save_to_parquet(batch, file_idx):
    df = pd.DataFrame({"text": batch})
    parquet_filename = f"fi_{file_idx}.parquet"
    df.to_parquet(parquet_filename, engine="pyarrow", index=False)
    print(f"Saved {len(batch)} lines to {parquet_filename}")

if __name__ == "__main__":
    process_files(INPUT_FOLDER)
