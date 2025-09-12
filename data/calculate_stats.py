import os
import re

# Folder containing the .log files
log_folder = "./logs"

# Regular expression to extract the number of tokens
token_pattern = re.compile(r"Stats: \{tokens: (\d+)")
total_tokens_before = 0
total_tokens_after = 0

# Iterate over all .log files in the folder
for filename in os.listdir(log_folder):
    if filename.endswith(".log"):
        file_path = os.path.join(log_folder, filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            matches = []
            for line in file:
                match = token_pattern.search(line)
                if match:
                    matches.append(match)  # Keep updating with the latest match

        # Process the last match for the file
        if matches != []:
            total_tokens_before += int(matches[0].group(1))
            total_tokens_after += int(matches[-1].group(1))

print(f"Total tokens across all files before cleaning: {total_tokens_before}")
print(f"Total tokens across all files after cleaning: {total_tokens_after}")
