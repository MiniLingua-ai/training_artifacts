#!/bin/bash
# SLURM job submission script for SFT training on a single node with 4 H200 GPUs.

#SBATCH --job-name=sft_equal                 # Human-readable job name (appears in queue)
#SBATCH --cpus-per-gpu=4                     # CPU threads allocated per GPU (for dataloading, etc.)
#SBATCH --nodes=1                            # Number of nodes to use
#SBATCH --gpus-per-node=4                    # Number of GPUs per node
#SBATCH --partition=gpu-h200-141g-short      # SLURM partition/queue to submit to
#SBATCH --mem=1000G                          # System RAM requested
#SBATCH --time=7:00:00                       # Max wall-clock time (HH:MM:SS)
#SBATCH -o /scratch/cs/small_lm/sft/train_logs/equally_balanced/1b_latest_lr_0.00002_constant_bs_256_shuffled_2_epochs/log.out  # Stdout log
#SBATCH -e /scratch/cs/small_lm/sft/train_logs/equally_balanced/1b_latest_lr_0.00002_constant_bs_256_shuffled_2_epochs/log.err  # Stderr log


export WORLD_SIZE=4                          # Total number of processes across all nodes (here: 4 GPUs -> 4 procs)

set -eox pipefail                            # Exit on error, print commands, fail on pipe errors
echo "Starting bash script"
module purge                                  # Reset all loaded modules (clean environment)


export SSL_CERT_FILE='/scratch/cs/small_lm/cacert.pem'  # Custom CA bundle if needed for HTTPS (e.g., HF/W&B)


export CUDA_VISIBLE_DEVICES=0,1,2,3          # Pin visible GPUs
export NCCL_P2P_DISABLE=0                    
export NCCL_IB_DISABLE=0                     



# -------------------------
# Training hyperparameters:
# MBS: per-GPU micro-batch size
# GBS: global batch size (across all GPUs and accumulation steps)
# LR: learning rate
# TOTAL_ITERS: total optimizer steps target (used as --train-iters). NOTE: when resuming from checkpoint, by default includes all iterations that elapsed previously while pre-training
# -------------------------
MBS="16"
GBS="256"
LR="0.00002"
TOTAL_ITERS="48_000_000"

# -------------------------
# Logging / saving / eval cadence:
# -------------------------
LOG_INTERVAL=1                               # Print logs every N iterations
SAVE_INTERVAL=500                            # Save checkpoint every N iterations
EVAL_INTERVAL=250                            # Run evaluation every N iterations
EVAL_STEPS=4                                 # Number of eval iterations per eval pass

CHECKPOINT_DIR="/scratch/cs/small_lm/sft/train_balanced/equally_balanced/1b_latest_lr_0.00002_constant_bs_256_shuffled_2_epochs_"  # Directory to load (resume) from

# Weights & Biases experiment configuration
WANDB_EXP_NAME="equally_balanced/1b_latest_lr_0.00002_constant_bs_256_shuffled_2_epochs"
WANDB_PROJECT="sft-small-lm"
WANDB_SAVE_DIR="wandb_logs"

# -------------------------
# Data paths:
# DATA_ROOT: base folder for jsonl datasets
# CACHE_PATH: optional index cache (speeds up loading)
# DATA_PATH: weighted list of training dataset jsonl files: "<weight> <file> ..."
# -------------------------
DATA_ROOT="/scratch/cs/small_lm/sft/sft_jsonl"
CACHE_PATH="${DATA_ROOT}/index-cache"

DATA_PATH="0.1 ${DATA_ROOT}/en.jsonl 0.1 ${DATA_ROOT}/es.jsonl 0.1 ${DATA_ROOT}/el.jsonl 0.1 ${DATA_ROOT}/pt.jsonl 0.1 ${DATA_ROOT}/pl.jsonl 0.1 ${DATA_ROOT}/fr.jsonl 0.1 ${DATA_ROOT}/fi.jsonl 0.1 ${DATA_ROOT}/sv.jsonl 0.1 ${DATA_ROOT}/it.jsonl 0.1  ${DATA_ROOT}/de.jsonl 0.1 ${DATA_ROOT}/nl.jsonl 0.1 ${DATA_ROOT}/cs.jsonl 0.1  ${DATA_ROOT}/bg.jsonl 0.1 ${DATA_ROOT}/code.jsonl"

# Output directory for new checkpoints (saves)
SAVE_PATH="/scratch/cs/small_lm/sft/train_balanced/${WANDB_EXP_NAME}"
mkdir -p $SAVE_PATH

# Tokenizer identifier (HF repo or local path)
TOKENIZER_MODEL="minilingua-ai/MiniLingua-1b-Instruct"


export CUDA_DEVICE_MAX_CONNECTIONS=1          # Limit CUDA connections (can help with sequence parallelism contention)


# -------------------------
# Distributed training basics:
# MASTER_ADDR/PORT: rendezvous for torch.distributed
# -------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)  # First node hostname
export MASTER_PORT=6001
export CUDA_DEVICE_MAX_CONNECTIONS=1 

# OMP threads used by each process (affects dataloader/ops)
export OMP_NUM_THREADS=4


# Reduce Python warnings verbosity in logs
export PYTHONWARNINGS=ignore

# TransformerEngine / ROCm related toggles (flash attention, arch hints)
export NVTE_FLASH_ATTN=1
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_ROCM_ARCH=gfx90a
export PYTORCH_ROCM_ARCH=gfx90a


# -------------------------
# Parse CLI overrides of the form KEY=VALUE and export them.
# Example: RECOMPUTATION=1 CP_SIZE=2 ./run.sh
# -------------------------
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# -------------------------
# Model & parallelism knobs:
# MODEL_SIZE: selects a config block below
# SEQ_LEN: max input sequence length
# RECOMPUTATION: 1 to enable activation checkpointing (saves memory)
# -------------------------
MODEL_SIZE="1B"
FSDP="0"
SEQ_LEN="2048"
RECOMPUTATION="${RECOMPUTATION:-0}"

# -------------------------
# Parallel args (Megatron-LM):
# PP: pipeline model parallel size
# TP: tensor model parallel size
# CP_SIZE: context-parallel size (sequence parallel complement)
# VPP: virtual pipeline stages (used if USE_VPP=1)
# USE_VPP: gate to enable virtual pipeline
# LOAD_CKPT_PATH / SAVE_CKPT_PATH: optional external checkpoint path overrides (unused here)
# PROFILE: 1 to enable PyTorch profiler
# -------------------------
PP="1"
TP="1"
CP_SIZE="${CP_SIZE:-1}"
VPP="${VPP:-1}"
USE_VPP="${USE_VPP:-0}"
LOAD_CKPT_PATH="${LOAD_CKPT_PATH:-None}"
SAVE_CKPT_PATH="${SAVE_CKPT_PATH:-None}"
PROFILE="${PROFILE:-0}"


# -------------------------
# Model architecture presets chosen by MODEL_SIZE
# Values mirror public SmolLM configs where noted.
# -------------------------
# https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json
if [[ $MODEL_SIZE = "200M" ]]; then #test
    NHIDDEN=576
    FFN_HIDDEN_SIZE=1536
    NLAYERS=30
    NHEADS=9
    NUM_KV_HEADS=3
    TIE_WORD_EMBEDDINGS=1
    
# https://huggingface.co/HuggingFaceTB/SmolLM-360M/blob/main/config.json
elif [ "$MODEL_SIZE" = "360M" ]; then
    NHIDDEN=960
    FFN_HIDDEN_SIZE=2560
    NLAYERS=32
    NHEADS=15
    NUM_KV_HEADS=5
    TIE_WORD_EMBEDDINGS=1

# https://huggingface.co/HuggingFaceTB/SmolLM-1.7B/blob/main/config.json
elif [ "$MODEL_SIZE" = "1B" ]; then
    NHIDDEN=1536
    FFN_HIDDEN_SIZE=6144
    NLAYERS=32
    NHEADS=24
    NUM_KV_HEADS=8
    TIE_WORD_EMBEDDINGS=1


elif [ "$MODEL_SIZE" = "30M" ]; then
    NHIDDEN=384
    FFN_HIDDEN_SIZE=1536
    NLAYERS=12
    NHEADS=6
    NUM_KV_HEADS=3
    TIE_WORD_EMBEDDINGS=1

elif [ "$MODEL_SIZE" = "60M" ]; then
    NHIDDEN=512
    FFN_HIDDEN_SIZE=2048
    NLAYERS=16
    NHEADS=8
    NUM_KV_HEADS=4
    TIE_WORD_EMBEDDINGS=1

elif [ "$MODEL_SIZE" = "100M" ]; then
    NHIDDEN=640
    FFN_HIDDEN_SIZE=2560
    NLAYERS=18
    NHEADS=10
    NUM_KV_HEADS=5
    TIE_WORD_EMBEDDINGS=1

else
    echo "Unknown model size"
    exit 1
fi


# -------------------------
# Build Megatron-LM argument strings incrementally:
# GPT_ARGS: model & training core settings
# -------------------------
GPT_ARGS="$GPT_ARGS --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NHEADS \
"
# Enable GQA if KV heads < attention heads
if [ "$NUM_KV_HEADS" != "$NHEADS" ]; then
    GPT_ARGS="$GPT_ARGS \
    --group-query-attention \
    --num-query-groups $NUM_KV_HEADS \
    "
fi

# Untie embeddings (LM head separate) if requested
if [ "$TIE_WORD_EMBEDDINGS" = "0" ]; then
    GPT_ARGS="$GPT_ARGS --untie-embeddings-and-output-weights \
    "
fi

# Choose parallelization backend: FSDP2 or Megatron TP/PP/CP
if [ "$FSDP" = "1" ]; then
    PARALLEL_ARGS="$PARALLEL_ARGS \
    --use-torch-fsdp2 \
    "
else
PARALLEL_ARGS="$PARALLEL_ARGS \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --context-parallel-size $CP_SIZE \
    --sequence-parallel \
    --use-distributed-optimizer \
    "
fi

# -------------------------
# Optional PyTorch profiler window (if PROFILE=1)
# -------------------------
#PYTORCH PROFILER ARGS
if [ "$PROFILE" = "1" ]; then
    PROFILE_ARGS="--use-pytorch-profiler --profile-ranks 0 --profile-step-start 5 --profile-step-end 7"
else
    PROFILE_ARGS=""
fi



# Core GPT class training options
GPT_ARGS="$GPT_ARGS \
    --load $CHECKPOINT_DIR \
    --use-checkpoint-args \
    --attention-softmax-in-fp32 \
    --max-position-embeddings $SEQ_LEN \
    --use-flash-attn \
    --seq-length $SEQ_LEN \
    --position-embedding-type rope \
    --rotary-base 10000 \
    --attention-dropout 0.1 \
    --hidden-dropout 0.1 \
    --normalization RMSNorm \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --train-iters $TOTAL_ITERS \
    --bf16 \
    --swiglu \
    --no-async-tensor-model-parallel-allreduce \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    --no-bias-dropout-fusion \
    --no-rope-fusion \
    --no-load-optim \
    --no-load-rng \
    --distributed-timeout-minutes 30 \
    --overlap-grad-reduce \
    --sft \
    --dist-ckpt-strictness ignore_all \
    "

# Optimizer & LR schedule
OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --ckpt-format torch \
    --lr $LR  \
    --lr-decay-style constant \
    --clip-grad 1.0 \
    --weight-decay 2.0e-2 \
    "


# Output/telemetry & split ratios
OUTPUT_ARGS=" \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters $EVAL_STEPS \
    --wandb-project $WANDB_PROJECT \
    --wandb-exp-name $WANDB_EXP_NAME \
    --wandb-save-dir $WANDB_SAVE_DIR \
    --log-throughput \
    --log-progress \
    --log-interval $LOG_INTERVAL \
    --split 97,2,1 \
    "


# Data loading & tokenizer
DATA_ARGS="\
    --tokenizer-type SFTTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --dataloader-type single \
    --num-workers 4 \
    --data-path $DATA_PATH \
"
# Enable virtual pipeline stages if requested
if [ "$USE_VPP" = "1" ]; then
    PARALLEL_ARGS="$PARALLEL_ARGS \
    --num-layers-per-virtual-pipeline-stage $VPP"
fi
# Activation recomputation to reduce memory
if [ "$RECOMPUTATION" = "1" ]; then
    GPT_ARGS="$GPT_ARGS --recompute-activations --recompute-granularity selective"
fi

# Checkpoint saving settings
CHECKPOINT_ARGS=""
CPKT_INTERVAL=1000                             # (Defined but not used; save cadence controlled by SAVE_INTERVAL)

CHECKPOINT_ARGS="$CHECKPOINT_ARGS \
    --save $SAVE_PATH \
    --save-interval $SAVE_INTERVAL \
    "


# -------------------------
# Final command to run inside the container:
# torchrun: launches one process per GPU (nproc_per_node=4)
# pretrain_gpt.py: Megatron-LM entrypoint
# -------------------------
CMD="torchrun \
    --nproc_per_node=4 \
    --master_addr $MASTER_ADDR \
    --master_port 6001 \
    /scratch/cs/small_lm/Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    $OPTIMIZER_ARGS \
    $PARALLEL_ARGS \
    $CHECKPOINT_ARGS \
    $OUTPUT_ARGS \
    $DATA_ARGS \
    $PROFILE_ARGS \
    "
echo '============='
echo $CMD
echo '============='


echo "START $SLURM_JOBID: $(date)"
echo "NNODES" $SLURM_NNODES
echo "CPUS PER TASK" $SLURM_CPUS_PER_GPU

# Singularity container image (must include all deps: PyTorch, Megatron, TE, etc.)
CONTAINER=/scratch/cs/small_lm/test.sif


# -------------------------
# srun + singularity:
# --nv: enable GPU passthrough
# -B fakelink: bind a fake CUDA compat lib path (cluster-specific workaround)
# --bind: mount host directories into container (/m, /l, /scratch)
# $CMD: runs the composed training command inside the container
# -------------------------
srun singularity exec \
    --nv \
    -B fakelink:/usr/local/cuda/compat/lib \
    --bind=/m,/l,/scratch \
    $CONTAINER \
    $CMD
