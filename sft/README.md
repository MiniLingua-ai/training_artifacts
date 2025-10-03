# MiniLingua 1B Supervised Fine-Tuning (SFT)

This folder contains scripts and configurations for supervised fine-tuning (SFT) of the MiniLingua 1B model. SFT adapts the pre-trained base model to follow multilingual and multi-domain instructions, making it more useful and better aligned with user needs.

## 🎯 Overview

Supervised Fine-Tuning is the second stage of model training where the goal is to adapt the model to follow instructions and perform specific tasks more effectively. The SFT process uses smaller, more carefully selected datasets compared to pre-training, focusing on instruction-response pairs across multiple languages and domains.

**Key Features**:
- **Multilingual Support**: 13 languages plus code instructions
- **Multi-Domain Coverage**: General QA, reasoning, knowledge, and coding tasks
- **Custom MCQA Dataset**: Specialized multiple-choice question answering
- **Output-Token-Only Training**: Efficient training focusing on response generation
- **Overfitting Prevention**: Careful monitoring and early stopping

## 📊 SFT Dataset Composition

### Core Multilingual Datasets

The SFT training uses a combination of established multilingual instruction datasets:

| Dataset | Languages | Description |
|---------|-----------|-------------|
| **BUFFET** | 54 | Multilingual benchmark adaptation dataset |
| **PolyglotPrompt** | 49 | Multilingual prompts adapted for benchmark tasks |
| **Bactrian-X** | 52 | Parallel multilingual dataset with 3.4M instruction-response pairs |
| **Aya** | 114 | Extensive multilingual collection of instruction-response pairs |
| **Alpaca** | 9 | Translated instruction-response pairs for supervised fine-tuning |

### Custom Multiple-Choice QA Dataset

A specialized dataset was created for multiple-choice question answering, supporting three answer formats:
- **Letter format** (40%): "A", "B", "C", "D"
- **Number format** (30%): "1", "2", "3", "4"  
- **Full answer format** (30%): Complete answer text

**Source Datasets for MCQA**:
- **ARC-Challenge**: Grade-school science questions
- **SweFAQ 2.0**: Swedish government FAQ dataset
- **Czech TruthfulQA**: Czech translation of TruthfulQA
- **INCLUDE**: Multilingual benchmark (44 languages) for knowledge and reasoning
- **EXAMS**: High school examination QA dataset with multilingual coverage

**Instruction Language Mix**:
- 30% instructions in English
- 70% instructions in target language

## 📁 Scripts Overview

### 1. `sft_train.sh` - Main Training Script
Executes the supervised fine-tuning process using Megatron-LM framework

**Key Configuration**:
- **Hardware**: 1 node, 4 NVIDIA H200 GPUs
- **Batch Size**: 256 instructions (16 micro-batch size)
- **Learning Rate**: 2e-6 (constant schedule)
- **Training Duration**: ~50 hours
- **Memory**: 1TB total memory allocation
- **Precision**: BFloat16 training


### 2. `read_data.ipynb` - Dataset Analysis and Processing
Comprehensive notebook for analyzing, processing, and cleaning multilingual SFT datasets

**What it does**:
- **Dataset Loading**: Reads various dataset formats (JSON, JSON.gz, Parquet)
- **Data Conversion**: Converts JSON/JSON.gz files to Parquet format with sampling
- **Token Analysis**: Counts tokens per instruction and output using the trained tokenizer
- **Language Detection**: Uses FastText to identify and filter by language
- **Data Cleaning**: Removes low-quality data and filters by sequence length
- **Format Standardization**: Applies consistent chat template formatting

**Chat Template Format**:
```
<|start_of_sequence|><|im_start|>
{instruction}
{input}
<|im_end|>
{output}<|end_of_sequence|>
```
**Quality Filters Applied**:
- Removes sequences longer than 2000 tokens
- Filters out incomplete instructions ending with ':'
- Excludes low-quality datasets (NQ-Open, etc.)
- Language verification using FastText model
- Dataset-specific quality controls

### 3. `generate_instructions.py` - JSONL Conversion
- Reads processed Parquet files for each language
- Applies Hugging Face chat template formatting
- Converts to conversation format with user/assistant roles
- Counts tokens using the trained tokenizer
- Generates token statistics per language

### 4. `data_shuffle.py` - Data Randomization
- Reads all JSONL files in the specified directory
- Randomly shuffles the order of instructions within each file
- Preserves the original file structure and format
- Ensures training data is presented in random order


### 5. `create_sft_mcqa.ipynb` - MCQA Dataset Creation
- Processes various QA datasets (ARC, SweFAQ, TruthfulQA, etc.)
- Generates instructions in both English and target languages
- Creates three answer formats (letter, number, full text)
- Balances language and format distributions
- Validates dataset quality and consistency

## 🔧 Training Configuration

### 🖥️ Hardware Setup
| Component | Specification |
|-----------|---------------|
| **Platform** | Aalto Triton Supercomputer |
| **Compute Nodes** | 1 node |
| **GPUs** | 4 × NVIDIA H200 (141GB each) |
| **Total Memory** | 1TB system memory |
| **Storage** | High-performance scratch storage |

### ⚙️ Training Parameters
```yaml
# 🎯 Core Training Settings
MICRO_BATCH_SIZE: 16
GLOBAL_BATCH_SIZE: 256
LEARNING_RATE: 2e-6
SEQUENCE_LENGTH: 2048
```


### 🌍 Language Distribution
| Language | Percentage |
|----------|------------|
| 🇬🇧 **English** | 27% |
| 🇩🇪 **German** | 18% |
| 🇪🇸 **Spanish** | 14% |
| 🇫🇷 **French** | 13% |
| 🇮🇹 **Italian** | 7% |
| 🇵🇹 **Portuguese** | 7% |
| 🇳🇱 **Dutch** | 4% |
| 🇵🇱 **Polish** | 4% |
| 🇨🇿 **Czech** | 2% |
| 🇧🇬 **Bulgarian** | 1% |
| 🇬🇷 **Greek** | 1% |
| 🇫🇮 **Finnish** | 1% |
| 🇸🇪 **Swedish** | 1% |
| 💻 **Code** | 5% |


### ⏱️ Training Performance
| Metric | Value |
|--------|-------|
| **⏰ Total Time** | ~50 hours |
| **🔄 Epochs** | 1 epoch (early stopping) |
| **👣 Total Steps** | ~6,000 steps |
| **⚡ Throughput** | ~43 instructions/second |
| **🖥️ Hardware** | 4 × NVIDIA H200 GPUs |
