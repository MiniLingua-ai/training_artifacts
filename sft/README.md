# Supervised Fine-Tuning (SFT)

This folder contains configuration files and notes for supervised fine-tuning (SFT) of the MiniLingua 1B model.  
SFT adapts the pretrained base model to follow multilingual and multi-domain instructions.  

## Contents

Training configs presented as `.sh` scripts include batch sizes, LR schedules, optimizer choices and other parameters as a part of Megatron-LM launch script.
`token_counts.csv` presents training data distribution.
`convert.sh` script was used to convert megatron model to HF-compatible format.
`training_loss.png` gives example of the training loss dynamics while going through SFT process.

## Training environment

We trained our models using [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM), running inside the official [NGC PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=25.04-py3).  

In addition, we installed [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) for optimized kernels and Hugging Face libraries (`transformers`, `datasets`, `tokenizers`) for data and tokenizer support. The versiones used are pinned in `requirements.txt`.

Setup follows directly from the instructions in the linked repositories, with minor environment variable adjustments needed specifically for the cluster we were working with. 