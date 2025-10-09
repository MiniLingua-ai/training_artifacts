#!/bin/bash
# SLURM Job Configuration
#SBATCH --job-name=1B_bs_1024_lr_0025_warmup_6000_start_0  # Job name identifying model size, batch size, learning rate, warmup steps
#SBATCH --cpus-per-task=7                                   # Number of CPU cores per task
#SBATCH --ntasks-per-node=8                                 # Number of tasks (GPUs) per node
#SBATCH --nodes=32                                          # Total number of compute nodes (32 nodes * 8 GPUs = 256 GPUs total)
#SBATCH --mem=480G                                          # Memory per node in GB
#SBATCH --partition=standard-g                              # SLURM partition for GPU jobs
#SBATCH --time=2-00:00:00                                   # Maximum job runtime (2 days)
#SBATCH --exclusive                                         # Exclusive access to nodes (no sharing with other jobs)
#SBATCH --gpus-per-node=8                                   # Number of GPUs per node
#SBATCH --account=project_462000756                         # SLURM account for billing/resource allocation
#SBATCH -o logs/%x-%j.out                                   # Standard output log file (%x=job name, %j=job ID)
#SBATCH -e logs/%x-%j.err                                   # Standard error log file

set -eox pipefail  # Exit on error, print commands, fail on pipe errors
echo "Starting bash script"
module purge
module load LUMI/24.03 partition/G  # Load LUMI supercomputer modules for GPU partition

# Core Training Hyperparameters
GBS="1024"        # Global Batch Size - total batch size across all GPUs
LR="0.0005"       # Learning Rate - step size for gradient updates
TOTAL_ITERS="725000"  # Total number of training iterations

# Logging and Checkpointing Configuration
LOG_INTERVAL=1        # Log training metrics every N iterations
SAVE_INTERVAL=1000    # Save model checkpoint every N iterations
EVAL_INTERVAL=1000    # Run evaluation every N iterations
EVAL_STEPS=4          # Number of evaluation steps to run


# Weights & Biases (W&B) Configuration for Experiment Tracking
WANDB_EXP_NAME="1B_bs_1024_lr_0025_warmup_6000_start_0"  # Experiment name for tracking
WANDB_PROJECT="small-lm"                                   # W&B project name
WANDB_SAVE_DIR="wandb_logs"                               # Directory to save W&B logs

# Data Paths - Different types of training data
WEB_DATA_ROOT="/scratch/project_462000756/data/train_web_bin"    # Web-scraped text data (multilingual)
CODE_DATA_ROOT="/scratch/project_462000756/data/code_bin"        # Source code data
HQ_DATA_ROOT="/scratch/project_462000756/data/train_hq_bin"      # High-quality curated text data
CACHE_PATH="${WEB_DATA_ROOT}/index-cache"                        # Cache for data loading optimization

# Data Mixing Configuration - Weighted sampling from different datasets
# Format: "weight path weight path ..." where weights sum to ~1.0
# Includes multilingual web data, high-quality text, and code data
DATA_PATH="0.2295 ${WEB_DATA_ROOT}/en 0.119 ${WEB_DATA_ROOT}/es 0.0085 ${WEB_DATA_ROOT}/el 0.0595 ${WEB_DATA_ROOT}/pt 0.034 ${WEB_DATA_ROOT}/pl 0.1105 ${WEB_DATA_ROOT}/fr 0.0085 ${WEB_DATA_ROOT}/fi 0.0085 ${WEB_DATA_ROOT}/sv 0.0595 ${WEB_DATA_ROOT}/it 0.153 ${WEB_DATA_ROOT}/de 0.034 ${WEB_DATA_ROOT}/nl 0.017 ${WEB_DATA_ROOT}/cs 0.0085 ${WEB_DATA_ROOT}/bg 0.027 ${HQ_DATA_ROOT}/en 0.014 ${HQ_DATA_ROOT}/es 0.001 ${HQ_DATA_ROOT}/el 0.007 ${HQ_DATA_ROOT}/pt 0.004 ${HQ_DATA_ROOT}/pl 0.013 ${HQ_DATA_ROOT}/fr 0.001 ${HQ_DATA_ROOT}/fi 0.001 ${HQ_DATA_ROOT}/sv 0.007 ${HQ_DATA_ROOT}/it 0.018 ${HQ_DATA_ROOT}/de 0.004 ${HQ_DATA_ROOT}/nl 0.002 ${HQ_DATA_ROOT}/cs 0.001 ${HQ_DATA_ROOT}/bg  0.05 ${CODE_DATA_ROOT}/code"

# Checkpoint Configuration
CHECKPOINT_PATH="/scratch/project_462000756/checkpoints"  # Base directory for saving model checkpoints
SAVE_PATH="${CHECKPOINT_PATH}/${WANDB_EXP_NAME}"          # Specific checkpoint directory for this experiment
mkdir -p $SAVE_PATH                                       # Create checkpoint directory

TOKENIZER_MODEL="/scratch/project_462000756/tokeniser/ConvertedTokenizer"  # Path to pre-trained tokenizer

# Create symbolic links for easy access to latest log files
ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.out logs/latest.out
ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.err logs/latest.err

# CUDA and Compiler Configuration
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Limit CUDA device connections for stability
export CC=gcc-12                      # C compiler version
export CXX=g++-12                     # C++ compiler version

# Distributed Training Configuration
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)  # Master node for distributed training
export MASTER_PORT=9999                # Port for distributed communication
export WORLD_SIZE=$SLURM_NTASKS        # Total number of processes (valid only if ntasks==ngpus)
export CUDA_DEVICE_MAX_CONNECTIONS=1   # Required for sequence parallelism

# OpenMP Threading Configuration
export OMP_NUM_THREADS=1               # Single thread per process to avoid oversubscription
export HSA_ENABLE_SDMA=0               # Disable HSA SDMA for AMD GPUs

# Debugging Configuration (mostly commented out for production)
# export MIOPEN_ENABLE_LOGGING=1          # Enable MIOpen logging for AMD GPUs
export PYTHONWARNINGS=ignore              # Suppress Python warnings for cleaner logs
# export TORCH_SHOW_CPP_STACKTRACES=1     # Show C++ stack traces in PyTorch errors
# export NCCL_DEBUG=INFO                  # Enable NCCL debugging information
# export RCCL_KERNEL_COLL_TRACE_ENABLE=1  # Enable RCCL kernel tracing
# export NCCL_DEBUG_SUBSYS=ALL            # Debug all NCCL subsystems
# export NCCL_DEBUG_FILE=nccl-debug/nccl-debug-${SLURM_JOB_NAME}-${SLURM_JOBID}.log  # Separate NCCL debug log

# TransformerEngine Configuration for Optimized Attention
export NVTE_FLASH_ATTN=1        # Enable FlashAttention for memory efficiency
export NVTE_DEBUG=0             # Disable TransformerEngine debugging
export NVTE_DEBUG_LEVEL=0       # Set debug level to 0 (no debug output)
export NVTE_ROCM_ARCH=gfx90a    # AMD GPU architecture for TransformerEngine
export PYTORCH_ROCM_ARCH=gfx90a # AMD GPU architecture for PyTorch

# Parse command line arguments and export as environment variables
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)
   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"
   export "$KEY"="$VALUE"
done

# Model Configuration
MODEL_SIZE="1B"                        # Model size (1B parameters)
FSDP="0"                              # Fully Sharded Data Parallel (0=disabled, 1=enabled)
SEQ_LEN="2048"                        # Maximum sequence length in tokens
MBS="4"                               # Micro Batch Size per GPU
RECOMPUTATION="${RECOMPUTATION:-0}"   # Activation recomputation for memory savings (0=off, 1=on)

# Parallelization Strategy
PP="1"                                # Pipeline Parallel size (number of pipeline stages)
TP="1"                                # Tensor Parallel size (intra-layer model parallelism)
CP_SIZE="${CP_SIZE:-1}"               # Context Parallel size (for long sequences)
VPP="${VPP:-1}"                       # Virtual Pipeline Parallel stages
USE_VPP="${USE_VPP:-0}"               # Enable Virtual Pipeline Parallelism (0=off, 1=on)

# Checkpoint and Profiling Configuration
# LOAD_CKPT_PATH=""                   # Path to checkpoint for resuming training (commented out)
SAVE_CKPT_PATH="${SAVE_CKPT_PATH:-None}"  # Path to save checkpoints
PROFILE="${PROFILE:-0}"               # Enable PyTorch profiler (0=off, 1=on)



# Model Architecture Configuration Based on Size
if [ "$MODEL_SIZE" = "1B" ]; then
    NHIDDEN=1536              # Hidden dimension size
    FFN_HIDDEN_SIZE=6144      # Feed-forward network hidden size (typically 4x hidden size)
    NLAYERS=32                # Number of transformer layers
    NHEADS=24                 # Number of attention heads
    NUM_KV_HEADS=8            # Number of key-value heads (for grouped query attention)
    TIE_WORD_EMBEDDINGS=1     # Tie input and output embeddings (1=tied, 0=separate)

elif [ "$MODEL_SIZE" = "30M" ]; then
    NHIDDEN=384               # Hidden dimension for 30M parameter model
    FFN_HIDDEN_SIZE=1536      # Feed-forward network size
    NLAYERS=12                # Number of layers
    NHEADS=6                  # Attention heads
    NUM_KV_HEADS=3            # Key-value heads
    TIE_WORD_EMBEDDINGS=1     # Tie embeddings

elif [ "$MODEL_SIZE" = "60M" ]; then
    NHIDDEN=512               # Hidden dimension for 60M parameter model
    FFN_HIDDEN_SIZE=2048      # Feed-forward network size
    NLAYERS=16                # Number of layers
    NHEADS=8                  # Attention heads
    NUM_KV_HEADS=4            # Key-value heads
    TIE_WORD_EMBEDDINGS=1     # Tie embeddings

elif [ "$MODEL_SIZE" = "100M" ]; then
    NHIDDEN=640               # Hidden dimension for 100M parameter model
    FFN_HIDDEN_SIZE=2560      # Feed-forward network size
    NLAYERS=18                # Number of layers
    NHEADS=10                 # Attention heads
    NUM_KV_HEADS=5            # Key-value heads
    TIE_WORD_EMBEDDINGS=1     # Tie embeddings

else
    echo "Unknown model size"
    exit 1
fi


# Build GPT Model Arguments
GPT_ARGS="$GPT_ARGS --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NHEADS \
"

# Enable Grouped Query Attention if KV heads differ from attention heads
if [ "$NUM_KV_HEADS" != "$NHEADS" ]; then
    GPT_ARGS="$GPT_ARGS \
    --group-query-attention \
    --num-query-groups $NUM_KV_HEADS \
    "
fi

# Untie embeddings if specified (separate input/output embedding weights)
if [ "$TIE_WORD_EMBEDDINGS" = "0" ]; then
    GPT_ARGS="$GPT_ARGS --untie-embeddings-and-output-weights \
    "
fi

# Configure Parallelization Strategy
if [ "$FSDP" = "1" ]; then
    # Use Fully Sharded Data Parallel (FSDP) for memory efficiency
    PARALLEL_ARGS="$PARALLEL_ARGS \
    --use-torch-fsdp2 \
    "
else
    # Use Megatron's parallelization strategies
    PARALLEL_ARGS="$PARALLEL_ARGS \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --context-parallel-size $CP_SIZE \
    --sequence-parallel \
    --use-distributed-optimizer \
    "
fi

# PyTorch Profiler Configuration
if [ "$PROFILE" = "1" ]; then
    PROFILE_ARGS="--use-pytorch-profiler --profile-ranks 0 --profile-step-start 5 --profile-step-end 7"
else
    PROFILE_ARGS=""
fi

# Core GPT Training Arguments
GPT_ARGS="$GPT_ARGS \
    --attention-softmax-in-fp32 \                    # Use FP32 for attention softmax (numerical stability)
    --max-position-embeddings $SEQ_LEN \             # Maximum sequence length for positional embeddings
    --use-flash-attn \                               # Use FlashAttention for memory efficiency
    --seq-length $SEQ_LEN \                          # Input sequence length
    --position-embedding-type rope \                 # Use RoPE (Rotary Position Embedding)
    --rotary-base 10000 \                           # Base frequency for RoPE
    --disable-bias-linear \                         # Disable bias in linear layers (modern practice)
    --init-method-std 0.02 \                        # Standard deviation for weight initialization
    --attention-dropout 0.0 \                       # Attention dropout rate (0.0 = no dropout)
    --hidden-dropout 0.0 \                          # Hidden layer dropout rate
    --normalization RMSNorm \                       # Use RMSNorm instead of LayerNorm
    --micro-batch-size $MBS \                       # Batch size per GPU
    --global-batch-size $GBS \                      # Total batch size across all GPUs
    --train-iters $TOTAL_ITERS \                    # Total number of training iterations
    --bf16 \                                        # Use bfloat16 precision for training
    --swiglu \                                      # Use SwiGLU activation function
    --no-async-tensor-model-parallel-allreduce \    # Disable async tensor parallel allreduce
    --no-masked-softmax-fusion \                    # Disable masked softmax fusion
    --no-gradient-accumulation-fusion \             # Disable gradient accumulation fusion
    --no-bias-dropout-fusion \                      # Disable bias dropout fusion
    --no-rope-fusion \                              # Disable RoPE fusion
    --no-load-optim \                               # Don't load optimizer state from checkpoint
    --no-load-rng \                                 # Don't load RNG state from checkpoint
    --distributed-timeout-minutes 30 \              # Timeout for distributed operations
    --overlap-grad-reduce \                         # Overlap gradient reduction with computation
    "

# Optimizer Configuration
OPTIMIZER_ARGS=" \
    --optimizer adam \                    # Use Adam optimizer
    --adam-beta1 0.9 \                   # Adam beta1 parameter (momentum)
    --adam-beta2 0.95 \                  # Adam beta2 parameter (RMSprop-like)
    --adam-eps 1e-8 \                    # Adam epsilon for numerical stability
    --ckpt-format torch \                # Save checkpoints in PyTorch format
    --lr $LR  \                          # Learning rate
    --lr-decay-style WSD \               # Warmup-Stable-Decay learning rate schedule
    --lr-wsd-decay-style linear \        # Linear decay during decay phase
    --lr-wsd-decay-iters 74000 \         # Number of iterations for decay phase
    --lr-decay-iters 74000 \             # Total decay iterations
    --clip-grad 1.0 \                    # Gradient clipping threshold
    --weight-decay 1.0e-2 \              # Weight decay (L2 regularization)
    --lr-warmup-iters 6000 \             # Number of warmup iterations
    "


# Output and Logging Configuration
OUTPUT_ARGS=" \
    --eval-interval $EVAL_INTERVAL \     # Run evaluation every N iterations
    --eval-iters $EVAL_STEPS \           # Number of evaluation steps
    --wandb-project $WANDB_PROJECT \     # Weights & Biases project name
    --wandb-exp-name $WANDB_EXP_NAME \   # Experiment name for W&B
    --wandb-save-dir $WANDB_SAVE_DIR \   # Directory to save W&B logs
    --log-throughput \                   # Log training throughput metrics
    --log-progress \                     # Log training progress
    --log-interval $LOG_INTERVAL \       # Log metrics every N iterations
    --split 98,1,1 \                     # Train/validation/test split percentages
    "

# Data Loading Configuration
DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \  # Use HuggingFace tokenizer
    --tokenizer-model ${TOKENIZER_MODEL} \   # Path to tokenizer model
    --dataloader-type single \               # Single dataloader type
    --num-workers 2 \                        # Number of data loading workers
    --data-path $DATA_PATH \                 # Weighted data paths
"
# Optional Features Configuration
if [ "$USE_VPP" = "1" ]; then
    # Enable Virtual Pipeline Parallelism for better memory usage
    PARALLEL_ARGS="$PARALLEL_ARGS \
    --num-layers-per-virtual-pipeline-stage $VPP"
fi

if [ "$RECOMPUTATION" = "1" ]; then
    # Enable activation recomputation to save memory at cost of compute
    GPT_ARGS="$GPT_ARGS --recompute-activations --recompute-granularity selective"
fi

# Checkpoint Configuration
CHECKPOINT_ARGS=""
CPKT_INTERVAL=1000  # Checkpoint save interval (unused variable)

CHECKPOINT_ARGS="$CHECKPOINT_ARGS \
    --save $SAVE_PATH \           # Directory to save checkpoints
    --save-interval $SAVE_INTERVAL \  # Save checkpoint every N iterations
    "

# Construct Final Training Command
CMD=" \
    Megatron-LM/pretrain_gpt.py \  # Main Megatron training script
    $GPT_ARGS \                    # Model architecture arguments
    $OPTIMIZER_ARGS \              # Optimizer configuration
    $PARALLEL_ARGS \               # Parallelization settings
    $CHECKPOINT_ARGS \             # Checkpoint configuration
    $OUTPUT_ARGS \                 # Logging and output settings
    $DATA_ARGS \                   # Data loading configuration
    $PROFILE_ARGS \                # Profiling arguments (if enabled)
    "
echo '============='
echo $CMD
echo '============='

# CPU Binding Configuration for LUMI Supercomputer
c="fe"  # Hexadecimal mask for CPU binding
# Bind mask for one thread per core to avoid oversubscription
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# Job Execution Information
echo "START $SLURM_JOBID: $(date)"
echo "NNODES" $SLURM_NNODES
echo "CPUS PER TASK" $SLURM_CPUS_PER_TASK

# Singularity Container Configuration for LUMI
CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif

# Bind mount paths for Singularity container access to LUMI filesystem
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd
export PWD=(`pwd -P`)  # Get absolute path

# Avoid conflicts with user's local Python packages
export PYTHONUSERBASE=""

launcher="$PWD/launcher.sh"  # Path to launcher script

# Execute Training Job with Proper CPU Binding
echo "Using --cpu-bind=mask_cpu:$BIND_MASK"
srun --label --cpu-bind=mask_cpu:$BIND_MASK \  # SLURM job launcher with CPU binding
    singularity exec \                          # Execute in Singularity container
    -B $PWD \                                   # Bind current directory
    $CONTAINER \                                # Container image
    $launcher \                                 # Launcher script
    $CMD                                        # Training command

echo "END $SLURM_JOBID: $(date)"

# Post-training throughput analysis
singularity exec -B $SINGULARITY_BIND $CONTAINER python3 tools/throughput.py logs/${SLURM_JOB_NAME}-${SLURM_JOBID}.out