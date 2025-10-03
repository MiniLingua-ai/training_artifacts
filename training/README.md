# MiniLingua 1B: Large Language Model Training

This repository contains the complete training pipeline for **MiniLingua 1B**, a 1-billion parameter decoder-only transformer language model trained from scratch using Megatron-LM.

## 🚀 Model Overview

**MiniLingua 1B** is a multilingual large language model with the following specifications:

- **Architecture**: Decoder-only Transformer (GPT-style)
- **Parameters**: 1 billion parameters
- **Context Length**: 2048 tokens
- **Training Framework**: Megatron-LM with PyTorch
- **Precision**: BFloat16 (bf16)
- **Languages**: Multilingual support (All languages and datasets are listed in the data folder)

### Model Architecture Details

```
Hidden Size: 1536
FFN Hidden Size: 6144
Number of Layers: 32
Attention Heads: 24
Key-Value Heads: 8 (Grouped Query Attention)
Position Embeddings: RoPE (Rotary Position Embedding)
Normalization: RMSNorm
Activation: SwiGLU
```

## 📁 Training Scripts

This folder contains training scripts that demonstrate different aspects of the complete training process:

### 1. Small-Scale Experiment (`small_experiment.sh`)

**Purpose**: Hyperparameter exploration and validation across multiple model sizes (30M, 60M, 100M parameters)

This script implements comprehensive hyperparameter optimization experiments to identify optimal learning rates and batch sizes for different model scales. The experiments follow scaling law principles to predict optimal hyperparameters for larger models.

#### Model Configurations Tested

| Model Name | Layers | Model Dim | FFN Dim | Attn Heads | KV Heads | Head Dim | Parameters |
|------------|--------|-----------|---------|------------|----------|----------|------------|
| MiniLingua-30M | 12 | 384 | 1536 | 6 | 3 | 64 | 26.5M |
| MiniLingua-60M | 16 | 512 | 2048 | 8 | 4 | 64 | 62.9M |
| MiniLingua-100M | 18 | 640 | 2560 | 10 | 5 | 64 | 110.6M |

#### Hyperparameter Grid Search

**Batch Sizes Tested**: 2^16, 2^17, 2^18, 2^19, 2^20, 2^21 tokens  
**Learning Rates Tested**: 0.0005, 0.001, 0.0025, 0.004, 0.005  
**Compute Budget**: Fixed FLOPs allocation per model size

| Model Size | Total FLOPs |
|------------|-------------|
| 30M | 5.2 × 10^18 |
| 60M | 1.8 × 10^19 |
| 110M | 5.4 × 10^19 |

#### Experimental Results

**30M Parameter Model - Validation Loss Results**:
| Learning Rate | 2^16 | 2^17 | 2^18 | 2^19 | 2^20 |
|---------------|------|------|------|------|------|
| 0.0005 | 3.78 | 3.64 | 3.59 | 3.58 | 3.62 |
| 0.001 | 3.77 | 3.63 | 3.57 | 3.55 | 3.56 |
| 0.0025 | 3.79 | 3.64 | 3.55 | **3.51** | 3.53 |
| 0.004 | 3.82 | 3.65 | 3.57 | 3.52 | 3.57 |
| 0.005 | 3.83 | 3.67 | 3.58 | 3.53 | 3.57 |

**60M Parameter Model - Validation Loss Results**:
| Learning Rate | 2^16 | 2^17 | 2^18 | 2^19 | 2^20 | 2^21 |
|---------------|------|------|------|------|------|------|
| 0.0005 | 3.51 | 3.38 | 3.32 | 3.30 | 3.32 | 3.37 |
| 0.001 | 3.52 | 3.38 | 3.32 | 3.28 | 3.28 | 3.30 |
| 0.0025 | 3.57 | 3.42 | 3.32 | 3.27 | **3.25** | 3.28 |
| 0.004 | 3.60 | 3.44 | 3.34 | 3.28 | 3.29 | 3.33 |
| 0.005 | 3.63 | 3.46 | 3.36 | 3.29 | 3.29 | 3.35 |

**110M Parameter Model - Validation Loss Results**:
| Learning Rate | 2^17 | 2^18 | 2^19 | 2^20 | 2^21 |
|---------------|------|------|------|------|------|
| 0.0005 | 3.21 | 3.14 | 3.11 | 3.11 | 3.15 |
| 0.001 | 3.22 | 3.14 | 3.10 | 3.10 | 3.12 |
| 0.0025 | 3.26 | 3.17 | 3.11 | **3.08** | 3.08 |
| 0.004 | 3.29 | 3.21 | 3.13 | 3.08 | 3.12 |
| 0.005 | 3.21 | 3.17 | 3.13 | 3.09 | 3.13 |

The experiments follow power law relationships for optimal hyperparameters:
```
α_opt = β × C^γ
```
Where α_opt is the optimal hyperparameter, β is the scaling coefficient, C is compute budget (FLOPs), and γ is the scaling exponent.

**Batch Size Scaling Results**:
- Scaling exponent (γ): 0.4285
- Scaling coefficient (β): 0.0047
- Optimal batch size for 1B model: ~12.5M tokens

**Learning Rate Scaling Results**:
- Scaling exponent (γ): -0.216
- Scaling coefficient (β): 33.1
- Optimal learning rate for 1B model: ~0.00059


### 2. Full-Scale Training (`full_train.sh`)

**Purpose**: Production training of the full 1B parameter MiniLingua model using a two-stage training approach

#### Training Configuration

**Core Setup**:
- **Model Architecture**: Llama 3 architecture with Grouped Query Attention (GQA)
- **Model Size**: 1 billion non-embedding parameters
- **Batch Size**: 2 million tokens
- **Total Training Steps**: 715,000
- **Maximum Sequence Length**: 2048 tokens
- **Hardware**: 32 compute nodes, 128 AMD MI200X GPUs (4 GPUs per node)
- **Training Duration**: ~12 days

**Technical Specifications**:
- **Position Embedding**: Rotary Position Embedding (RoPE) with base frequency 10,000
- **Activation Function**: Swish-Gated Linear Unit (SwiGLU)
- **Precision**: BFloat16 training
- **Warmup Steps**: 6,000 iterations
- **Learning Rate Scheduler**: Warmup-Stable-Decay (WSD)
- **Decay Style**: Linear decay
- **Decay Iterations**: 71,500 (10% of total training steps)

#### Two-Stage Training Approach

**Stage 1: Constant Learning Rate Phase (0-643,500 steps)**
- **Learning Rate**: Initially it was 0.0025, then it was reduced to 0.0005 after plateau detection
- **Data Focus**: General multilingual content

**Stage 2: Learning Rate Decay Phase (643,500-715,000 steps)**
- **Learning Rate**: Linear decrease from 0.0005 to 0.
- **Data Focus**: Higher-quality data and refined language proportions


#### Dataset Composition by Stage

**Stage 1**:
- Web Data: 85%
- High-Quality Data: 10%
- Code Data: 5%

**Stage 2**:
- Web Data: 60%
- High-Quality Data: 30%
- Code Data: 10%

#### Language Distribution Strategy

The training employed different language distribution strategies, with Stage 2 testing three approaches:

**Stage 1 Distribution**:
- English: 27%, German: 18%, Spanish: 14%, French: 13%
- Italian: 7%, Portuguese: 7%, Dutch: 4%, Polish: 4%
- Czech: 2%, Greek: 1%, Finnish: 1%, Swedish: 1%, Bulgarian: 1%

**Stage 2 - Increased Low-Resource Allocation** (Selected):
- English: 20%, German: 14%, Spanish: 12%, French: 12%
- Italian: 7%, Portuguese: 7%, Dutch: 5%, Polish: 5%
- Czech: 4%, Bulgarian: 4%, Finnish: 4%, Greek: 3%, Swedish: 3%

This distribution was chosen to improve performance in underrepresented languages while maintaining strong representation for high-resource languages.

### 3. Checkpoint Conversion (`convert.sh`)
Convert Megatron checkpoints to HuggingFace format for inference and deployment