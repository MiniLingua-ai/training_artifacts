import sentencepiece as spm
import os
import pandas as pd
import wandb
import psutil
import argparse
from collections import defaultdict
from tqdm.auto import tqdm
from multiprocessing import Pool

def run_processing(file):
    file_path_to_id = defaultdict(set)
    df = pd.read_parquet(file)
    
    for _, row in tqdm(df.iterrows()):
        doc_id = int(row["id"].strip().rsplit('/', 1)[1])
        doc_path = row["metadata"]["file_path"]
        file_path_to_id[doc_path].add(doc_id)
    
    # Log CPU and RAM usage
    wandb.log({
        "CPU_Usage": psutil.cpu_percent(),
        "RAM_Usage": psutil.virtual_memory().percent
    })
    
    return file_path_to_id

def generator(folder="./../test_web"):
    global_file_path_to_id = defaultdict(set)
    
    for root, _, files in os.walk(folder):
        if len(files) > 0:
            pathes = [os.path.join(root, file_) for file_ in files]
            with Pool(16) as p:
                processed_data = p.map(run_processing, pathes)
            for data in processed_data:
                for key, vals in data.items():
                    global_file_path_to_id[key].update(vals)
                
    for file_path, indices in global_file_path_to_id.items():
        df = pd.read_parquet(file_path)
        for idx_ in indices:
            text = df.iloc[idx_]["text"].strip()
            if len(text) == 0:
                continue
            
            # Log system memory usage while yielding
            wandb.log({
                "RAM_Usage": psutil.virtual_memory().percent
            })
            
            yield text
        
def get_file_names(folder_name):
    files = os.listdir(f"./../tokeniser_scripts/{folder_name}")
    files = [os.path.join(f"./../tokeniser_scripts/{folder_name}", f) for f in files]
    return ','.join(files)

def train_tokenizer(args):
    wandb.login(key="key")
    wandb.init(project="Tokenizer training", name=f"tokenizer_{args.model_type}_{args.vocab_size}_tsv", config={
        "vocab_size": args.vocab_size,
        "model_type": args.model_type,
        "folder_name": args.folder_name
    }, dir="./wandb_common")
    
    files = get_file_names(args.folder_name)

    spm.SentencePieceTrainer.train(
        input=files,
        input_format="tsv",
        model_prefix=f"tokenizer_{args.folder_name}_{args.model_type}_{args.vocab_size}",
        vocab_size=args.vocab_size,
        split_digits=True, 
        byte_fallback=True,
        remove_extra_whitespaces=False,
        train_extremely_large_corpus=True,
        model_type=args.model_type,
        # split_by_whtespace=True
        # input_sentence_size=100000,
        # max_sentence_length=16768,

    )
    
    # Log final CPU and RAM usage
    wandb.log({
        "Final_CPU_Usage": psutil.cpu_percent(),
        "Final_RAM_Usage": psutil.virtual_memory().percent
    })
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer with flexible parameters.")
    
    parser.add_argument("--vocab_size", type=int, default=50000, help="Vocabulary size.")
    parser.add_argument("--model_type", type=str, choices=["unigram", "bpe", "word", "char"], default="bpe",
                        help="Type of SentencePiece model (unigram, bpe, word, char).")
    
    parser.add_argument("--folder_name", type=str, help="Folder name with tsv files.")

    args = parser.parse_args()
    train_tokenizer(args)
