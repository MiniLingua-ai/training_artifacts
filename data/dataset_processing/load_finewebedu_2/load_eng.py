# Example script for loading English dataset.
import os
from datasets import load_dataset, DownloadConfig, Dataset
from huggingface_hub import login
from tqdm import tqdm
import logging
from settings import SETTINGS

os.environ['HF_HOME'] = '/scratch/cs/small_lm/cache/'

config = DownloadConfig(
    cache_dir="/scratch/cs/small_lm/fine_web_edu_2",
    extract_on_the_fly=True,
    delete_extracted=True,
    num_proc=20
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='culturax_eng.log',
    filemode='w'
)

login(SETTINGS["hf"]["token"])

logging.info(f"Starting loading eng")
ds = load_dataset("HuggingFaceFW/fineweb",
                  name="sample-350BT",
                  split="train",
                  cache_dir="/scratch/cs/small_lm/fine_web_edu_2",
                  keep_in_memory=False,
                  download_config=config,
                  streaming=True)

dir_name = f"/scratch/cs/small_lm/fine_web_edu_2/eng"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

batch_size = 1e5
batch = []
full_dataset = []
initial_number = 0
file_name = "web-edu-eng-train-{num}.parquet"
last_file = None

for ex in tqdm(ds):
    batch.append({"text": ex["text"]})
    if len(batch) >= batch_size:
        full_dataset.extend(batch)
        batch = []
        if len(full_dataset) > 5e5:
            logging.info(f"Processed file number {initial_number}")
            tmp_dataset = Dataset.from_list(full_dataset)
            tmp_dataset.to_parquet(os.path.join(dir_name, file_name.format(lang="en", num=initial_number)))
            full_dataset = []
            initial_number += 1

if len(batch) > 0:
    full_dataset.extend(batch)
    tmp_dataset = Dataset.from_list(full_dataset)
    tmp_dataset.to_parquet(os.path.join(dir_name, file_name.format(lang="en", num=initial_number)))

logging.info("eng is loaded.")
        