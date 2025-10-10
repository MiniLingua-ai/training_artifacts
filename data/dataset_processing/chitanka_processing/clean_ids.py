import argparse
import re
import pandas as pd
import os


parser = argparse.ArgumentParser()
parser.add_argument("idx", type=int, help="Index for file selection")
args = parser.parse_args()

# Get list of files
files = os.listdir(f"/scratch/cs/small_lm/chitanka/")
files = [os.path.join("/scratch/cs/small_lm/chitanka", f) for f in files if f"bg_{args.idx}_" in f]

def clean_text(text):
    # Remove everything from "$id = " onwards
    text = re.split(r"\$id\s*=", text, maxsplit=1)[0]
    # Strip leading/trailing whitespace
    return text.strip()

for file in files:
    df = pd.read_parquet(file)
    df["text"] = df["text"].apply(clean_text)
    df.to_parquet(file)

print(f'Processed idx {args.idx} and {len(files)} files')