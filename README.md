# MiniLingua – Training Artifacts

This repository contains the training artifacts of the **MiniLingua** model – a small multilingual language model with 1B parameters, trained on 13 European languages and code.  

## 🤗 Model Access

The final trained models and tokenizer are available on Hugging Face:

**[https://huggingface.co/minilingua-ai](https://huggingface.co/minilingua-ai)**


Quick start with `Transformers` both for GPU and CPU enabled envs:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "minilingua-ai/MiniLingua-1b-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
gen = pipeline("text-generation", model=model, tokenizer=tokenizer, trust_remote_code=True)

prompt = "Translate from Bulgarian: Здравейте! Как сте? Translation:"
out = gen(prompt, max_new_tokens=128, do_sample=False)
print(out[0])
```

**Note:** since the instruct model was trained in `bfloat16` using `cuda 3.12` (see exact nvidia image in `sft` folder), `vllm`-powered inference is available only on devices that support respective library versions, i.e. A100 and newer.

---

## Repository Structure

The repository is organized as follows:

- **`data/`**  
  Contains scripts and notes related to dataset preparation, filtering, and preprocessing across 13 languages and code datasets.  

- **`evaluation/`**  
  Includes evaluation scripts, metrics, and benchmark setup.

- **`sft/`**  
  Contains artifacts related to supervised fine-tuning (SFT), including configuration files and guidance for applying instruction tuning.  

- **`tokenizer/`**  
  Holds tokenizer training artifacts. Includes scripts for training Byte-Pair Encoding (BPE) tokenizers and preparing multilingual vocab.  

- **`training/`**  
  Contains training scripts, and setup details for running large-scale distributed training of the MiniLingua model.  

---

## Notes on Training Environment

The training of MiniLingua was conducted on two large-scale HPC clusters:  

- **[LUMI Supercomputer (CSC Finland)](https://lumi-supercomputer.eu)**  
  Used for large-batch pretraining runs across multiple nodes.  

- **[Triton Supercomputer (Aalto University / CSC Finland)](https://scicomp.aalto.fi/triton/)**  
  Used for experimental runs, tokenizer pretraining, and SFT experiments.  

Each folder contains setup instructions and environment hints for reproducing the training on these clusters.  

---

## Authors:

- **Anna Aksenova**: aaaksenova2@gmail.com
- **Boris Zverkov**: bzorientbz@gmail.com 


