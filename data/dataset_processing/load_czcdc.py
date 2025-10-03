import os
from tqdm import tqdm

data = []
count = 0

for root, _, files in os.walk("./CzCDC/"):
    if 'SupAdmCo' in root:
        encode_ = "utf-8"
    else:
        encode_ = "Windows-1250"
    for f in tqdm(files):
        try:
            with open(os.path.join(root, f), 'r', encoding=encode_) as file:
                text = file.read().strip()
                count += len(text.split())
                data.append({"text": text})
        except:
            print("wrong encoding", f)


from datasets import Dataset


for num, i in enumerate(range(0, len(data), 100000)):
    df = Dataset.from_list(data[i:i+100000])
    df.to_parquet(f"./CzCDC/cs_{num}.parquet")
