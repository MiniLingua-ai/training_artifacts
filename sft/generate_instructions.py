import os
import glob
import pandas as pd
import json
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

# List of languages
languages = ["es", "en", "de", "fr", "fi", "pl", "pt", "it", "bg", "cs", "el", "nl", "sv", "code"]

# Paths
input_folder = "/scratch/cs/small_lm/sft/cleaned"
output_folder = os.path.join(input_folder, "sft_jsonl_qa_format")
os.makedirs(output_folder, exist_ok=True)

# Load tokenizer with chat_template
tokenizer = PreTrainedTokenizerFast.from_pretrained("/scratch/cs/small_lm/ConvertedTokenizer")
assert tokenizer.chat_template is not None, "Tokenizer must have a chat_template"

# Track token stats
token_counts = []

for lang in languages:
    pattern = os.path.join(input_folder, f"*_{lang}.parquet")
    parquet_files = glob.glob(pattern)

    if not parquet_files:
        print(f"⚠️ No files found for language: {lang}")
        continue

    output_path = os.path.join(output_folder, f"{lang}.jsonl")
    total_rows = 0
    total_tokens = 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
            except Exception as e:
                print(f"❌ Error reading {parquet_file}: {e}")
                continue

            for _, row in df.iterrows():
                instruction = str(row["instruction"]).strip()
                input_text = str(row["input"]).strip()
                output = str(row["output"]).strip()

                user_content = f"{instruction}\n{input_text}" if input_text else instruction

                # Format into Hugging Face-compatible dict
                chat_turn = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": output}
                    ]
                }

                f_out.write(json.dumps(chat_turn, ensure_ascii=False) + "\n")
                total_rows += 1

                # Token counting using chat template
                try:
                    rendered = tokenizer.apply_chat_template(chat_turn["messages"], tokenize=False)
                    tokens = tokenizer(rendered, return_tensors="pt").input_ids.shape[-1]
                    total_tokens += tokens
                except Exception as e:
                    print(f"⚠️ Tokenizer failed on {lang}: {e}")

    print(f"✅ {lang}: Wrote {total_rows} messages | {total_tokens} tokens → {output_path}")
    token_counts.append({"language": lang, "messages": total_rows, "tokens": total_tokens})

# Save token statistics
df_stats = pd.DataFrame(token_counts)
df_stats.to_csv(os.path.join(input_folder, "token_counts_qa_format.csv"), index=False)
print("📄 Saved token stats to token_counts.csv")
