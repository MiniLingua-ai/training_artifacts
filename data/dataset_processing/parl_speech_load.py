# import pyreadr
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

df = pd.read_csv("./parl_speech_v2/tweedekamer.csv")
df.head()

df = df.dropna()
text_data = []
count = 0
for i in tqdm(df.values):
    text_data.append({"text": i[0]})
    count += len(i[0].split())

del df
df = Dataset.from_list(text_data)
df.to_parquet("./parl_speech_v2/nl.parquet")