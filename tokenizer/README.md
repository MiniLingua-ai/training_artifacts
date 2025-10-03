# MiniLingua 1B Tokenizer

This folder contains scripts and tools for creating and evaluating the multilingual SentencePiece tokenizer used in MiniLingua 1B. The tokenizer was trained jointly across 13 languages plus code to ensure consistent vocabulary and efficient multilingual representation.

## 🎯 Overview

The MiniLingua tokenizer is a **SentencePiece BPE (Byte-Pair Encoding)** tokenizer designed for multilingual language modeling. It supports:

- **13 Languages**: English, German, Spanish, French, Italian, Portuguese, Dutch, Polish, Czech, Bulgarian, Greek, Finnish, Swedish
- **Code Support**: 9 programming languages (Python, C, JSON, XML, SQL, Markdown, YAML, TeX, Shell)
- **Vocabulary Size**: 128,000 tokens
- **Special Features**: Byte fallback, digit splitting, multilingual optimization

## 📁 Scripts Overview

### 1. `create_tsv.py` - Dataset Preparation
- Reads multilingual text data from Parquet files
- Applies language-specific tokenization patterns (regex-based)
- Balances data according to predefined language distributions
- Creates word frequency files in TSV format for SentencePiece training

**Key Features**:
- **Language Distribution Control**: Configurable proportions for each language
- **Code Data Integration**: Handles 9 different programming languages
- **Memory Efficient**: Processes data in batches to manage memory usage
- **Regex Tokenization**: Language-specific patterns for better text splitting


**Language Distributions Supported**:
- Training distribution (used in final model)
- Balanced distribution (equal representation)
- Original FineWeb distribution
- Custom distributions for different experiments

### 2. `tokeniser_training.py` - Tokenizer Training

- Trains SentencePiece BPE tokenizer on multilingual data
- Supports different model types (BPE, Unigram, Word, Char)
- Logs training progress to Weights & Biases

### 3. `tokeniser_eval.py` - Tokenizer Evaluation
- Tests tokenizer on multilingual text samples
- Counts tokens per language to analyze efficiency
- Generates language-specific tokenization statistics
- Creates evaluation reports in TSV format

### 4. `tokenise_training_data.py` - Data Tokenization
- Applies the trained tokenizer to training datasets
- Processes data in parallel using SLURM job scheduler (500 tasks with 64 workers each)
- Converts text to token sequences for model training
- Handles large datasets efficiently with distributed processing


**Usage**:
```bash
python tokenise_training_data.py --data_path train_balanced --lang bg
```

**Configuration**:
- **Tasks**: 500 parallel tasks
- **Workers**: 64 workers per task
- **Time Limit**: 12 hours
- **Memory**: 20GB per CPU
- **Partition**: batch-milan (SLURM partition)


### Language-Specific Regex Patterns
- **English**: Handles contractions (I'll, don't, etc.)
- **German**: Optimized for compound words
- **Italian/French**: Special handling for apostrophes and contractions
- **Other Languages**: General Unicode pattern for consistent tokenization

This tokenizer setup ensures optimal multilingual performance for the MiniLingua 1B model while maintaining efficiency and consistency across all supported languages.