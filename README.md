# MiniLingua – Training Artifacts

This repository contains the training artifacts of the **MiniLingua** model – a small multilingual language model with 1B parameters, trained on 13 European languages and code.  

## 🤗 Model Access

The final trained models and tokenizer are available on Hugging Face:

**[https://huggingface.co/minilingua-ai](https://huggingface.co/minilingua-ai)**

---

## Repository Structure

The repository is organized as follows:

- **`data/`**  
  Contains scripts and notes related to dataset preparation, filtering, and preprocessing across 13 languages and multilingual code datasets.  

- **`evaluation/`**  
  Includes evaluation scripts, metrics, and benchmark setup instructions.

- **`sft/`**  
  Contains artifacts related to supervised fine-tuning (SFT), including configuration files and guidance for applying instruction tuning.  

- **`tokenizer/`**  
  Holds tokenizer training artifacts and vocabularies. Includes scripts for training Byte-Pair Encoding (BPE) tokenizers and preparing multilingual vocab.  

- **`training/`**  
  Contains training logs, scripts, and setup details for running large-scale distributed training of the MiniLingua model.  

---

## Notes on Training Environment

The training of MiniLingua was conducted on two large-scale HPC clusters:  

- **LUMI Supercomputer (CSC Finland)**  
  Used for large-batch pretraining runs across multiple nodes.  

- **Triton Supercomputer (Aalto University / CSC Finland)**  
  Used for experimental runs, tokenizer pretraining, and SFT experiments.  

Each folder contains setup instructions and environment hints for reproducing the training on these clusters.  

---

## Authors:

- **Anna Aksenova**: aaaksenova2@gmail.com
- **Boris Zverkov**: bzorientbz@gmail.com 


