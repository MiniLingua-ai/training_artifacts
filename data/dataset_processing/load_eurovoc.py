import os
import pandas as pd

langs = ['bg', 'cs', 'nl', 'de', 'el', 'it', 'pl', 'fi', 'es', 'pt', 'fr', 'sv']

# Directory containing the text files
input_folder = "eurovoc"

# Loop over each language
for lang in langs:
    file_path = os.path.join(input_folder, f"{lang}.txt")
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Filter lines longer than 2 words
        # lines = [line.strip() for line in lines if len(line.split()) > 2]

        # Create DataFrame
        df = pd.DataFrame(lines, columns=['text'])
        del lines

        # Save as Parquet file
        output_path = os.path.join(input_folder, f"{lang}.parquet")
        df.to_parquet(output_path, index=False)
        print(f"Saved {lang}.parquet with {len(df)} rows.")
    else:
        print(f"File {file_path} does not exist.")