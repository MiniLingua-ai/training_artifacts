# Example script for cleaning English lines from a dataset.
from fasttext.FastText import _FastText
from datatrove.io import cached_asset_path_or_download
import pandas as pd
import argparse
import os
import numpy as np


model_file = cached_asset_path_or_download(
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    namespace="lid",
    subfolder="ft176",
    desc="fast-text language identifier model",
)
model = _FastText(model_file)


def clean_english(text):
    lines = text.split('\n')
    new_text = []
    for line in lines:
        if len(line) < 3:
            new_text.append(line)
            continue
        if model.predict(line)[0][0] == '__label__en':
            return True
    #         continue
    #     new_text.append(line)
    # new_text = '\n'.join(new_text)
    return False


parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path for file selection")
args = parser.parse_args()

files = os.listdir(args.path)
files = [os.path.join(args.path, f) for f in files if f"parquet" in f]

for file in files:
    df = pd.read_parquet(file)
    df['en'] = df['text'].apply(clean_english)
    count = np.sum(df['en'].values)
    per = count / df.shape[0]
    print(f'In document {file} there are {per} of texts with English lines.')