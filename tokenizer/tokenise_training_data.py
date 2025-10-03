import argparse
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers.parquet import ParquetReaderProxy, ParquetReader
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
import json

# Set up argument parser
parser = argparse.ArgumentParser(description="Process training data.")
parser.add_argument("--data_path", type=str, default="train_balanced", help="Path to training data")
parser.add_argument("--lang", type=str, default="bg", help="Language code")

# Parse arguments
args = parser.parse_args()

data_path = args.data_path
lang = args.lang


dist_executor = SlurmPipelineExecutor(
    job_name=f"{lang}_{data_path}",
    pipeline=[
        ParquetReader(
            f"/scratch/cs/small_lm/{data_path}",  # read directly from huggingface
            # data_folder=f"/scratch/cs/small_lm/{data_path}",
            glob_pattern=f"*.parquet", 
        ),
        DocumentTokenizer(
            output_folder=f"/scratch/cs/small_lm/tokenise_all_data/{data_path}/tokenised/{lang}",
            local_working_dir=f"/scratch/cs/small_lm/tokenise_all_data/tmp/{data_path}/{lang}",
            save_filename=f"{data_path}_tokenized",
            eos_token="<|end_of_sequence|>",  # whether to add the EOS token after each document
            tokenizer_name_or_path="/scratch/cs/small_lm/ConvertedTokenizer/tokenizer.json",  
        ),
    ],
    tasks=500,
    workers=64,
    time="12:00:00",
    partition="batch-milan",
    mem_per_cpu_gb=20,
    cpus_per_task=5,
    logging_dir=f"/scratch/cs/small_lm/tokenise_all_data/tmp/logs/{data_path}/{lang}",
)
dist_executor.run()
