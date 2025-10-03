# Example script for cleaning English lines from a dataset.
from fasttext.FastText import _FastText
from datatrove.io import cached_asset_path_or_download
import pandas as pd
import argparse
import os
import re


model_file = cached_asset_path_or_download(
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    namespace="lid",
    subfolder="ft176",
    desc="fast-text language identifier model",
)
model = _FastText(model_file)


def is_only_numbers_or_whitespace(line):
    return bool(re.fullmatch(r"[0-9 ]*", line))


def clean_english(text):
    lines = text.split('\n')
    new_text = []
    for line in lines:
        # delete lines that are just numbers
        if is_only_numbers_or_whitespace(line):
            continue
        # don't check English for short lnes
        if len(line) < 3:
            new_text.append(line)
            continue
        # Delete lines with tweets
        if ':' in line and ('@' in line or '#' in line):
            continue
        # Delete lines with tweets
        # if len(line.strip().split(':')) > 4:
        #     continue
        # Delete English lines
        if model.predict(line)[0][0] == '__label__en':
            continue
        new_text.append(line)
    new_text = '\n'.join(new_text)
    return new_text


parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path for file selection")
args = parser.parse_args()

files = os.listdir(args.path)
files = [os.path.join(args.path, f) for f in files if f"parquet" in f]

for file in files:
    df = pd.read_parquet(file)
    df['text'] = df['text'].apply(clean_english)
    df.to_parquet(file)
    print(f'Processed document {file}.')
