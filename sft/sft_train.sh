#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --cpus-per-gpu=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu-h200-141g-short 
#SBATCH --mem=1000G
#SBATCH --time=30:00:00
#SBATCH -o /scratch/cs/small_lm/sft/train_logs/train_balanced/1b_latest_lr_0.000002_constant_bs_256_shuffled_2_epochs/log.out
#SBATCH -e /scratch/cs/small_lm/sft/train_logs/train_balanced/1b_latest_lr_0.00002_constant_bs_256_shuffled_2_epochs/log.err

export WORLD_SIZE=4

set -eox pipefail
echo "Starting bash script"
module purge


export SSL_CERT_FILE='/scratch/cs/small_lm/cacert.pem'


export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0



MBS="16"
GBS="256"
LR="0.000002"
TOTAL_ITERS="48_000_000" # total number of steps was controlled manually during training

#SAVING AND EVAL
LOG_INTERVAL=1
SAVE_INTERVAL=500
EVAL_INTERVAL=250
EVAL_STEPS=4

CHECKPOINT_DIR="/scratch/cs/small_lm/sft/train_balanced/train_balanced/1b_latest_lr_0.00002_constant_bs_256_shuffled_2_epochs"
# "/scratch/cs/small_lm/converted_checkpoints/final_llm_megatron"

WANDB_EXP_NAME="train_balanced/1b_latest_lr_0.000002_constant_bs_256_shuffled_2_epochs"


WANDB_PROJECT="sft-small-lm"
WANDB_SAVE_DIR="wandb_logs"

DATA_ROOT="/scratch/cs/small_lm/sft/sft_jsonl"
CACHE_PATH="${DATA_ROOT}/index-cache"

DATA_PATH="0.27 ${DATA_ROOT}/en.jsonl 0.14 ${DATA_ROOT}/es.jsonl 0.01 ${DATA_ROOT}/el.jsonl 0.07 ${DATA_ROOT}/pt.jsonl 0.04 ${DATA_ROOT}/pl.jsonl 0.13 ${DATA_ROOT}/fr.jsonl 0.01 ${DATA_ROOT}/fi.jsonl 0.01 ${DATA_ROOT}/sv.jsonl 0.07 ${DATA_ROOT}/it.jsonl 0.18  ${DATA_ROOT}/de.jsonl 0.04 ${DATA_ROOT}/nl.jsonl 0.02 ${DATA_ROOT}/cs.jsonl 0.01  ${DATA_ROOT}/bg.jsonl 0.05 ${DATA_ROOT}/code.jsonl"


SAVE_PATH="/scratch/cs/small_lm/sft/train_balanced/${WANDB_EXP_NAME}"
mkdir -p $SAVE_PATH

TOKENIZER_MODEL="aaaksenova/small_llm_eu"

# ln -sf "${SLURM_JOB_NAME}-${SLURM_JOBID}.out" logs/latest.out
# ln -sf "${SLURM_JOB_NAME}-${SLURM_JOBID}.err" logs/latest.err

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export CC=gcc-12
# export CXX=g++-12

#DISTRIBUTED ARGS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6001
export CUDA_DEVICE_MAX_CONNECTIONS=1 #This is needed for sequence paralellism

#OMP THREADING
export OMP_NUM_THREADS=4


#DEBUGGING, INCREASE VERBOSITY IN LOGS
# export MIOPEN_ENABLE_LOGGING=1
export PYTHONWARNINGS=ignore
# export TORCH_SHOW_CPP_STACKTRACES=1 
# export NCCL_DEBUG=INFO
# export RCCL_KERNEL_COLL_TRACE_ENABLE=1 
# export NCCL_DEBUG_SUBSYS=ALL 
# export NCCL_DEBUG_FILE=nccl-debug/nccl-debug-${SLURM_JOB_NAME}-${SLURM_JOBID}.log #Move verbose nccl logging to its own file

#TransformerEngine
export NVTE_FLASH_ATTN=1
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_ROCM_ARCH=gfx90a
export PYTORCH_ROCM_ARCH=gfx90a
# export TORCH_LOGS="+dynamo" 
# export TORCHDYNAMO_VERBOSE=1

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

MODEL_SIZE="1B"
FSDP="0"
SEQ_LEN="2048"
RECOMPUTATION="${RECOMPUTATION:-0}"

# PARALLEL ARGS
PP="1"
TP="1"
CP_SIZE="${CP_SIZE:-1}"
VPP="${VPP:-1}"
USE_VPP="${USE_VPP:-0}"
LOAD_CKPT_PATH="${LOAD_CKPT_PATH:-None}"
SAVE_CKPT_PATH="${SAVE_CKPT_PATH:-None}"
PROFILE="${PROFILE:-0}"


if [ "$MODEL_SIZE" = "1B" ]; then
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


GPT_ARGS="$GPT_ARGS --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NHEADS \
"
if [ "$NUM_KV_HEADS" != "$NHEADS" ]; then
    GPT_ARGS="$GPT_ARGS \
    --group-query-attention \
    --num-query-groups $NUM_KV_HEADS \
    "
fi

if [ "$TIE_WORD_EMBEDDINGS" = "0" ]; then
    GPT_ARGS="$GPT_ARGS --untie-embeddings-and-output-weights \
    "
fi

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

#TRAINING ARGS
#PYTORCH PROFILER ARGS
if [ "$PROFILE" = "1" ]; then
    PROFILE_ARGS="--use-pytorch-profiler --profile-ranks 0 --profile-step-start 5 --profile-step-end 7"
else
    PROFILE_ARGS=""
fi


# --tensor-model-parallel-size ${TP} \
# --pipeline-model-parallel-size 1 \
# --seq-length 4096 \
# --max-position-embeddings 4096 \
# --tokenizer-type HuggingFaceTokenizer \
# --tokenizer-model ${TOKENIZER_MODEL} \
# --load ${CHECKPOINT_DIR} \
# --exit-on-missing-checkpoint \
# --use-checkpoint-args \
# --no-load-optim \
# --no-load-rng \
# --untie-embeddings-and-output-weights \
# --normalization RMSNorm \
# --position-embedding-type rope \
# --no-masked-softmax-fusion \
# --attention-softmax-in-fp32
# --apply-layernorm-1p \
# --transformer-impl transformer_engine \
# --group-query-attention 8 \
# --disable-bia-linear \
# --rotary-base 1000000 \
# --rotary-percent 1.0 \
# --swiglu \
# --ffn-hidden-size 14336 \
# --num-attention-heads 32


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

#TENSORBOARD_PATH="tensorboard/$SLURM_JOB_NAME"
#OUTPUT_ARGS=" \
#    --eval-interval $EVAL_INTERVAL \
#    --eval-iters $EVAL_STEPS \
#    --tensorboard-dir $TENSORBOARD_PATH \
#    --tensorboard-queue-size 5 \
#    --log-throughput \
#    --log-progress \
#    --log-interval $LOG_INTERVAL \
#    "

DATA_ARGS="
    --tokenizer-type SFTTokenizer \
    --tokenizer-model aaaksenova/small_llm_eu \
    --dataloader-type single \
    --num-workers 4 \
    --data-path $DATA_PATH \
"
if [ "$USE_VPP" = "1" ]; then
    PARALLEL_ARGS="$PARALLEL_ARGS \
    --num-layers-per-virtual-pipeline-stage $VPP"
fi
if [ "$RECOMPUTATION" = "1" ]; then
    GPT_ARGS="$GPT_ARGS --recompute-activations --recompute-granularity selective"
fi

CHECKPOINT_ARGS=""
CPKT_INTERVAL=1000

CHECKPOINT_ARGS="$CHECKPOINT_ARGS \
    --save $SAVE_PATH \
    --save-interval $SAVE_INTERVAL \
    "


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

CONTAINER=/scratch/cs/small_lm/test.sif

# export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd
# export PWD=(`pwd -P`)

# # Avoid conflicts with $HOME/.local
# export PYTHONUSERBASE=""

# launcher="$PWD/launcher.sh"

# echo "Using --cpu-bind=mask_cpu:$BIND_MASK"
# unset BIND_MASK

srun singularity exec \
    --nv \
    -B fakelink:/usr/local/cuda/compat/lib \
    --bind=/m,/l,/scratch \
    $CONTAINER \
    $CMD

