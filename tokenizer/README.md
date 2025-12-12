# MiniLingua 1B Tokenizer

This folder contains scripts and tools for creating and evaluating the multilingual SentencePiece tokenizer used in MiniLingua 1B. The tokenizer was trained jointly across 13 languages plus code to ensure consistent vocabulary and efficient multilingual representation.

## Overview

The MiniLingua tokenizer is a **SentencePiece BPE (Byte-Pair Encoding)** tokenizer designed for multilingual language modeling. It supports:

- **13 Languages**: English, German, Spanish, French, Italian, Portuguese, Dutch, Polish, Czech, Bulgarian, Greek, Finnish, Swedish
- **Code Support**: 9 programming languages (Python, C, JSON, XML, SQL, Markdown, YAML, TeX, Shell)
- **Vocabulary Size**: 128,000 tokens
- **Special Features**: Byte fallback, digit splitting, multilingual optimization

## Tokeniser training

The training was done using `sentencepiece==0.2.0` and `tokenizers==0.21.2`. To set up the training and overcome memory bottle-neck we trained tokenisers on pre-tokenised text with token frequencies. 

1. `create_tsv.py` runs over pre-processed `parquet` files with pre-set sampled per-language weights and calculates pretokenised expressions. Make sure to create train and test files with the same weights this way.
2. `tokeniser_training.py` runs `SentencePieceTrainer` to train on pre-counted tokens.
3. `tokeniser_eval.py` evaluates the compression rate of the tokenisers against the test dataset.

## Performance Evaluation

Our tokenizer shows better word compression compared to existing multilingual tokenizers. The evaluation uses **Normalized Sequence Length (NSL)** - a metric that measures tokenization efficiency where lower values indicate better compression and fewer tokens needed to represent the same text.

### NSL Comparison Results

![NSL Without Code](avg_nsl.png)

The first chart shows NSL performance across different vocabulary sizes, comparing our tokenizer variants (Balanced, Original, Train, Intermediate, High Quality) against GPT-4o and EuroLLM. Our **Balanced-128k** tokenizer consistently achieves the lowest NSL scores, indicating superior tokenization efficiency.

![NSL Scores by Language](nsl_per_lang.png)

The second chart demonstrates language-specific performance across 13 languages. Our tokenizer shows competitive or superior performance across all tested languages, with particularly strong results in:
- **Finnish (fi)** and **Greek (el)**: Significant improvements over baseline models
- **Germanic languages** (de, nl, sv): Consistent efficiency gains
- **Romance languages** (fr, es, it, pt): Balanced performance across the language family

**Key Findings**:
- **25-30% better efficiency** compared to GPT-4o on average
- **15-20% improvement** over EuroLLM across most languages
- **Consistent performance** across all 13 supported languages
- **Optimal vocabulary size** at 128k tokens for multilingual scenarios