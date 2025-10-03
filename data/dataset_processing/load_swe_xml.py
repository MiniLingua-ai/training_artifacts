from datasets import Dataset
from tqdm.auto import tqdm
import logging
import html

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='swe_xml.log',
    filemode='w'
)

# some of them have <w tokens, some of them have <token tokens
pathes = ["./biblioteksbladet/runeberg-biblblad.xml", "./svt22/svt-2022.xml", "./humanoria/sweachum.xml",
          "./akademiska/sweacsam.xml", "./poems/poeter.xml", "./sv_literature/lb-open.xml"
          ]




for path in pathes:
    logging.info(f"Starting loading {path}")
    tokens_count = 0
    data_texts = []
    current_text = ""
    with open(path, "r") as file:
        for num, line in tqdm(enumerate(file)):
            if "<paragraph" in line:
                current_text += "\n"
            elif '<ne' in line or "</ne" in line:
                continue
            elif '</text' in line:
                current_text = current_text.strip().replace(" .", ".").replace(" , ", ", ").replace(" - ", "-").replace("- ", '').replace("-", " - ").replace(" % ", "% ").replace(" : ", ": ").replace(" ; ", "; ").replace(" !", "!").replace(" ?", "?").replace(" ) ", ") ").replace(" ( ", " (").replace(" / ", "/ ").replace(" ' ", "'").replace(' " ', '"')
                current_text = html.unescape(current_text)
                tokens_count += len(current_text.split())
                data_texts.append({"text": current_text})
                current_text = ""
                continue
            elif "<text" in line:
                assert current_text == "", f"Current text is not empty: {current_text}"
                if 'title="' in line:
                    title = line.split('title="')[1].split('" url=')[0]
                    current_text += title + "\n"
            elif '<token' in line:
                current_text += line.rsplit('">', 1)[1].split('<')[0]
                try:
                    end_token = line.split('_tail="')[1][:2]
                    end_token = end_token.replace("\\s", " ").replace("\t", "    ")
                except:
                    end_token = " "
                current_text += end_token
            elif '<w' in line:
                current_text += line.rsplit('">', 1)[1].split('</')[0] + " "
    logging.info(f"Tokens count: {tokens_count}")
    df = Dataset.from_list(data_texts)
    df.to_parquet(f'{path.rsplit("/", 1)[0]}/sv.parquet')
    logging.info(f"Saved to {path.rsplit('/', 1)[0]}/sv.parquet")