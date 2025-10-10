import os
import json
import random

# Path to the folder containing the .jsonl files
folder_path = '/scratch/cs/small_lm/sft/sft_jsonl'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jsonl'):
        file_path = os.path.join(folder_path, filename)

        # Read the data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]

        # Shuffle the data
        random.shuffle(data)

        # Write the shuffled data back
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f'Shuffled and updated: {filename}')
