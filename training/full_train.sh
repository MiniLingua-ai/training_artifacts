#!/bin/bash
#SBATCH --job-name=1B_bs_1024_lr_0025_warmup_6000_start_0
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=32
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000756
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -eox pipefail
echo "Starting bash script"
module purge
module load LUMI/24.03 partition/G

GBS="1024"
LR="0.0005"
TOTAL_ITERS="725000"

#SAVING AND EVAL
LOG_INTERVAL=1
SAVE_INTERVAL=1000
EVAL_INTERVAL=1000
EVAL_STEPS=4


WANDB_EXP_NAME="1B_bs_1024_lr_0025_warmup_6000_start_0"


WANDB_PROJECT="small-lm"
WANDB_SAVE_DIR="wandb_logs"

WEB_DATA_ROOT="/scratch/project_462000756/data/train_web_bin"
CODE_DATA_ROOT="/scratch/project_462000756/data/code_bin"
HQ_DATA_ROOT="/scratch/project_462000756/data/train_hq_bin"
CACHE_PATH="${WEB_DATA_ROOT}/index-cache"

DATA_PATH="0.2295 ${WEB_DATA_ROOT}/en 0.119 ${WEB_DATA_ROOT}/es 0.0085 ${WEB_DATA_ROOT}/el 0.0595 ${WEB_DATA_ROOT}/pt 0.034 ${WEB_DATA_ROOT}/pl 0.1105 ${WEB_DATA_ROOT}/fr 0.0085 ${WEB_DATA_ROOT}/fi 0.0085 ${WEB_DATA_ROOT}/sv 0.0595 ${WEB_DATA_ROOT}/it 0.153 ${WEB_DATA_ROOT}/de 0.034 ${WEB_DATA_ROOT}/nl 0.017 ${WEB_DATA_ROOT}/cs 0.0085 ${WEB_DATA_ROOT}/bg 0.027 ${HQ_DATA_ROOT}/en 0.014 ${HQ_DATA_ROOT}/es 0.001 ${HQ_DATA_ROOT}/el 0.007 ${HQ_DATA_ROOT}/pt 0.004 ${HQ_DATA_ROOT}/pl 0.013 ${HQ_DATA_ROOT}/fr 0.001 ${HQ_DATA_ROOT}/fi 0.001 ${HQ_DATA_ROOT}/sv 0.007 ${HQ_DATA_ROOT}/it 0.018 ${HQ_DATA_ROOT}/de 0.004 ${HQ_DATA_ROOT}/nl 0.002 ${HQ_DATA_ROOT}/cs 0.001 ${HQ_DATA_ROOT}/bg  0.05 ${CODE_DATA_ROOT}/code"


CHECKPOINT_PATH = "/scratch/project_462000756/checkpoints"
SAVE_PATH="${CHECKPOINT_PATH}/${WANDB_EXP_NAME}"
mkdir -p $SAVE_PATH

TOKENIZER_MODEL="/scratch/project_462000756/tokeniser/ConvertedTokenizer"

ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.out logs/latest.out
ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.err logs/latest.err

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CC=gcc-12
export CXX=g++-12

#DISTRIBUTED ARGS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS #This is valid only if ntasks==ngpus
export CUDA_DEVICE_MAX_CONNECTIONS=1 #This is needed for sequence paralellism

#OMP THREADING
export OMP_NUM_THREADS=1
export HSA_ENABLE_SDMA=0

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
MBS="4"
RECOMPUTATION="${RECOMPUTATION:-0}"

# PARALLEL ARGS
PP="1"
TP="1"
CP_SIZE="${CP_SIZE:-1}"
VPP="${VPP:-1}"
USE_VPP="${USE_VPP:-0}"

# LOAD_CKPT_PATH="" # Checkpoint path to continue training from
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

GPT_ARGS="$GPT_ARGS \
    --attention-softmax-in-fp32 \
    --max-position-embeddings $SEQ_LEN \
    --use-flash-attn \
    --seq-length $SEQ_LEN \
    --position-embedding-type rope \
    --rotary-base 10000 \
    --disable-bias-linear \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
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
    "

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --ckpt-format torch \
    --lr $LR  \
    --lr-decay-style WSD \
    --lr-wsd-decay-style linear \
    --lr-wsd-decay-iters 74000 \
    --lr-decay-iters 74000
    --clip-grad 1.0 \
    --weight-decay 1.0e-2 \
    --lr-warmup-iters 6000 \
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
    --split 98,1,1 \
    "

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type single \
    --num-workers 2 \
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

CMD=" \
    Megatron-LM/pretrain_gpt.py \
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


c="fe"
# Bind mask for one thread per core
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

echo "START $SLURM_JOBID: $(date)"
echo "NNODES" $SLURM_NNODES
echo "CPUS PER TASK" $SLURM_CPUS_PER_TASK

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif

export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd
export PWD=(`pwd -P`)

# Avoid conflicts with $HOME/.local
export PYTHONUSERBASE=""

launcher="$PWD/launcher.sh"

echo "Using --cpu-bind=mask_cpu:$BIND_MASK"
srun --label --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    $launcher \
    $CMD

echo "END $SLURM_JOBID: $(date)"

singularity exec -B $SINGULARITY_BIND $CONTAINER python3 tools/throughput.py logs/${SLURM_JOB_NAME}-${SLURM_JOBID}.out