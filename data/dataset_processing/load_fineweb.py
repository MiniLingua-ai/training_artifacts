# import os
# os.environ['HF_HOME'] = '/scratch/cs/small_lm/cache/'
# from datasets import load_dataset, DownloadConfig, Dataset
# from huggingface_hub import login
# from tqdm import tqdm
# import logging


# config = DownloadConfig(
#     cache_dir="/scratch/cs/small_lm/culturax",
#     extract_on_the_fly=True,
#     delete_extracted=True,
#     num_proc=20
#     )


# logging.basicConfig(
#     level=logging.INFO,  # Set logging level
#     format='%(asctime)s - %(levelname)s - %(message)s',  # Format of log messages
#     filename='culturax.log',  # Log messages to a file
#     filemode='w'  # Overwrite the log file; use 'a' to append
# )

# from settings import SETTINGS

# login(SETTINGS["hf"]["token"])


# # Load culturaX dataset

# langs = ["bg", "cs", "de", "el", "fi", "es", 
#           "fr", "it", "nl", "pl", "pt", "sv"]

# # logging.info("Starting loading English")
# # ds = load_dataset("uonlp/CulturaX",
# #                     "ba",
# #                     split="train",
# #                     keep_in_memory=False,
# #                     download_config=config
# #                   )
# # logging.info("English is loaded")

# for lang in langs:
#     initial_number = 0
#     file_name = "cultura-x-{lang}-train-{num}.arrow"
#     logging.info(f"Starting loading {lang}")
#     ds = load_dataset("uonlp/CulturaX",
#                     lang,
#                     split="train",
#                     cache_dir="/scratch/cs/small_lm/culturax",
#                     keep_in_memory=False,
#                     download_config=config,
#                     streaming=True
#                   )
#     dir_name = f"/scratch/cs/small_lm/culturax/uonlp___cultura_x/{lang}"
#     if not os.path.exists(dir_name):
#         os.makedirs(dir_name)
#     batch_size = 100000
#     batch = []
#     full_dataset = []
#     last_file = None
#     for ex in tqdm(ds):
#         batch.append(ex)
#         if len(batch) >= batch_size:
#             full_dataset.extend(batch)
#             batch = []
#             file_num = len(os.listdir('/scratch/cs/small_lm/cache/hub/datasets--uonlp--CulturaX/blobs'))
#             if file_num == 1 and last_file is None:
#                 last_file = os.listdir('/scratch/cs/small_lm/cache/hub/datasets--uonlp--CulturaX/blobs')[0]
#             elif file_num > 1:
#                 os.system(f'rm -rf /scratch/cs/small_lm/cache/hub/datasets--uonlp--CulturaX/blobs/{last_file}')
#                 logging.info(f"Processed file number {initial_number}")
#                 last_file = os.listdir('/scratch/cs/small_lm/cache/hub/datasets--uonlp--CulturaX/blobs')[0]
#                 tmp_dataset = Dataset.from_list(full_dataset)
#                 tmp_dataset.save_to_disk(os.path.join(dir_name, file_name.format(lang=lang, num=initial_number)))
#                 full_dataset = []
#                 initial_number += 1
#             if len(full_dataset) > 1e6:
#                 logging.info(f"Processed file number {initial_number}")
#                 tmp_dataset = Dataset.from_list(full_dataset)
#                 tmp_dataset.save_to_disk(os.path.join(dir_name, file_name.format(lang=lang, num=initial_number)))
#                 full_dataset = []
#                 initial_number += 1
#     if len(batch) > 0:
#         full_dataset.extend(batch)
#         tmp_dataset = Dataset.from_list(full_dataset)
#         tmp_dataset.save_to_disk(os.path.join(dir_name, file_name.format(lang=lang, num=initial_number)))
#     logging.info(f"{lang} is loaded.")
#     # logging.info(f"Size of downloads folder: {len(os.listdir('/scratch/cs/small_lm/culturax/downloads'))}")
#     # os.system('rm -rf /scratch/cs/small_lm/culturax/downloads')
#     # logging.info(f"Size of blobs folder: {len(os.listdir('/scratch/cs/small_lm/cache/hub/datasets--uonlp--CulturaX/blobs'))}")
#     # os.system('rm -rf /scratch/cs/small_lm/cache/hub/datasets---uonlp-CulturaX/blobs')
#     # os.system('rm -rf /scratch/cs/small_lm/cache')



import os
from datasets import load_dataset, DownloadConfig, Dataset
from huggingface_hub import login
from tqdm import tqdm
import logging
from settings import SETTINGS

os.environ['HF_HOME'] = '/scratch/cs/small_lm/cache/'

config = DownloadConfig(
    cache_dir="/scratch/cs/small_lm/culturax",
    extract_on_the_fly=True,
    delete_extracted=True,
    num_proc=20
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='culturax.log',
    filemode='w'
)

login(SETTINGS["hf"]["token"])

short_langs = ["bg", "cs", "de", "el", "fi", "es", 
         "fr", "it", "nl", "pl", "pt", "sv"]

langs = ["bul_Cyrl", "ces_Latn", "deu_Latn", "ell_Grek", "fin_Latn", "fra_Latn", 
         "spa_Latn", "ita_Latn", "nld_Latn", "pol_Latn", "por_Latn", "swe_Latn"]

for num, lang in enumerate(langs):
    short_lang = short_langs[num]
    with open(f"./load_finewebedu_2/load_{lang}.py", "w") as f:
        f.write(f"""
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
    filename='culturax_{lang}.log',
    filemode='w'
)

login(SETTINGS["hf"]["token"])

logging.info(f"Starting loading {lang}")
ds = load_dataset("HuggingFaceFW/fineweb-2",
                  name="{lang}",
                  split="train",
                  cache_dir="/scratch/cs/small_lm/fine_web_edu_2",
                  keep_in_memory=False,
                  download_config=config,
                  streaming=True)

dir_name = f"/scratch/cs/small_lm/fine_web_edu_2/{lang}"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

batch_size = 1e5
batch = []
full_dataset = []
initial_number = 0
file_name = "web-edu-{lang}-train-{{num}}.parquet"
last_file = None

for ex in tqdm(ds):
    batch.append({{"text": ex["text"]}})
    if len(batch) >= batch_size:
        full_dataset.extend(batch)
        batch = []
        if len(full_dataset) > 5e5:
            logging.info(f"Processed file number {{initial_number}}")
            tmp_dataset = Dataset.from_list(full_dataset)
            tmp_dataset.to_parquet(os.path.join(dir_name, file_name.format(lang="{short_lang}", num=initial_number)))
            full_dataset = []
            initial_number += 1

if len(batch) > 0:
    full_dataset.extend(batch)
    tmp_dataset = Dataset.from_list(full_dataset)
    tmp_dataset.to_parquet(os.path.join(dir_name, file_name.format(lang="{short_lang}", num=initial_number)))

logging.info("{lang} is loaded.")
        """)
