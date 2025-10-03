# MiniLingua 1B Evaluation

This folder contains evaluation scripts and benchmarks for assessing the **MiniLingua 1B** model's performance across multiple languages and tasks. The evaluation covers both base and instruction-tuned checkpoints using established multilingual benchmarks.

## 🎯 Evaluation Overview

The evaluation framework assesses MiniLingua 1B across diverse multilingual tasks.

## 📊 Evaluation Benchmarks

### Core Benchmarks
| Benchmark | Task Type | Languages | Response Format | Description | Evaluation script |
|-----------|-----------|-----------|-----------------|-------------|-----------------|
| **FLORES-200** | Translation | 13 | Free text | English-to-target translation | `flores_eval.py` |
| **Belebele** | Reading Comprehension | 13 | Multiple choice | Open-book question answering | `belebele_llm_judge.py` |
| **SIB** | Classification | 13 | Multiple choice | Topic classification | `sib_eval.py` |
| **MMLU** | Knowledge | 13 | Multiple choice | Multitask language understanding | `mmlu_eval.py` |
| **MassiveSum** | Summarization | 12 | Free text | Text summarization | `massivesum_eval.py` |

### Multiple-Choice Format Support
For multiple-choice tasks (Belebele, SIB, MMLU), the model is evaluated with:
- **Letter format**: A, B, C, D responses
- **Number format**: 1, 2, 3, 4 responses

## 📈 Performance Results

### Base Model Performance
**MiniLingua-1b-base vs. Competing Models**

| Model | FLORES*↑ | Belebele↑ | SIB↑ | MMLU↑ |
|-------|----------|-----------|------|-------|
| **MiniLingua-1b-base** | 0.343 | 0.23 | **0.248** | 0.24 |
| Gemma-3-1b-pt | 0.319 | 0.22 | 0.23 | 0.20 |
| EuroLLM-1.7b | *0.36* | *0.25* | 0.23 | *0.25* |
| Smollm2-1.7b | **0.367** | **0.32** | *0.24* | **0.32** |

*Note: Despite being trained on fewer tokens, MiniLingua-1b-base performs competitively across all benchmarks.*

### SFT Model Performance (English Instructions)

| Model | FLORES↑ | Belebele↑ |  | SIB↑ |  | MMLU↑ |  | MSum↑ |
|-------|---------|-----------|--|------|--|-------|--|-------|
|  |  | Letter | Number | Letter | Number | Letter | Number |  |
| **MiniLingua-1b** | *0.681* | 0.262 | 0.262 | 0.149 | 0.146 | 0.245 | 0.255 | **0.187** |
| Gemma-3-1b-it | 0.494 | *0.366* | *0.362* | *0.558* | *0.589* | **0.311** | **0.282** | 0.001 |
| EuroLLM-1.7b-Instruct | **0.879** | 0.216 | 0.218 | 0.124 | 0.116 | 0.205 | 0.225 | 0.0138 |
| Smollm2-1.7b-Instruct | 0.496 | **0.369** | **0.386** | **0.613** | **0.617** | *0.271* | *0.280* | *0.015* |

### SFT Model Performance (Native Language Instructions)

| Model | FLORES↑ | Belebele↑ | SIB↑ | MMLU↑ | MSum↑ |
|-------|---------|-----------|------|-------|-------|
| **MiniLingua-1b** | *0.759* | 0.274 | 0.127 | *0.250* | **0.174** |
| Gemma-3-1b-it | 0.494 | **0.335** | **0.455** | 0.246 | 0.0014 |
| EuroLLM-1.7b-Instruct | **0.835** | 0.270 | 0.135 | 0.235 | 0.013 |
| Smollm2-1.7b-Instruct | 0.486 | *0.330* | *0.349* | **0.266** | *0.015* |


This evaluation framework provides comprehensive assessment of MiniLingua 1B's multilingual capabilities across diverse NLP tasks, demonstrating competitive performance despite efficient training.