import os
import re
import lzma


def process_document(file_path, output_dir):
    """
    Process the document line by line and write the extracted text to separate files.

    :param file_path: Path to the input file
    :param output_dir: Directory to save the output files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_doc = None
    token_count = 0
    current_words = []
    doc_count = 0

    with lzma.open(file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            # Detect new document
            if line.startswith("<doc"):
                # current_doc_ = re.search(r'title="(.*?)"', line)
                # print(f'processing {current_doc_}')

                if current_doc is not None:
                    if doc_count // 50 == 0:
                        print(f"Processed {doc_count} documents")
                    # Write the previous document to a file
                    save_path = os.path.join(output_dir, f"{doc_count}.txt")
                    text = " ".join(current_words)
                    text = re.sub(r'\s+([,\.!?:;])', r'\1', text)
                    text = text.replace(' "', '"')
                    text = text.replace('" ', '"')
                    text = text.replace(' )', ')')
                    text = text.replace('( ', '(')
                    with open(save_path, 'w', encoding='utf-8') as output_file:
                        output_file.write(text)
                    
                    doc_count += 1

                    token_count += len(text.split())
                    print(f'Processed tokens: {token_count}')

                    # Start a new document
                current_doc = re.search(r'title="(.*?)"', line)
                print(f'processing {current_doc}')
                current_blocks = []
                current_words = []

            # Handle block (paragraphs)
            elif line.startswith("</block>"):
                #print('found block end')
                if current_words:
                    current_words.append("\n")  # Separate paragraphs with a newline

            elif line.startswith("<"):
                continue

            else:
                # we are at the normal words
                word = line.split()[0]
                current_words.append(word)

    # Write the last document if it exists
    if current_doc is not None:
        save_path = os.path.join(output_dir, f"{doc_count}.txt")
        text = " ".join(current_words)
        text = re.sub(r'\s+([,\.!?:;])', r'\1', text)
        text = text.replace(' "', '"')
        text = text.replace('" ', '"')
        text = text.replace(' )', ')')
        text = text.replace('( ', '(')
        with open(save_path, 'w', encoding='utf-8') as output_file:
            output_file.write(text)

    print(f"Processed {doc_count} documents")


# Usage example
input_file = "syn_v4.xz"
output_directory = "cs_syn_corpus"
print('Job started')
process_document(input_file, output_directory)
