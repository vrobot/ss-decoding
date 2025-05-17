#!/usr/bin/env bash
# run_sweep.sh  –  one-liner pipeline: dump → fit → parallel eval
# edit the variables section only

### ---- CONFIG -----------------------------------------------------------
# near the top of run_sweep.sh
PY=/mnt/ss-decoding/anil/.venv/bin/python      # absolute venv interpreter
MODEL_ID="meta-llama/Llama-4-Scout-17B-16E"
DATA_FILE="/mnt/ss-decoding/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
N_PROMPTS=50000            # train rows
SEQ=256                    # trunc length
BATCH=16                   # Batch size for dump_all.py
LAYER_STEP=4               # evaluate every 4th layer
GPUS=$(nvidia-smi -L | wc -l)   # auto-detect
NUM_LAYERS=48              # Set to the total number of layers in your model
ACT_DIR="acts"
HEAD_DIR="heads"
LOG_DIR="logs"
FIT_LAMBDA=1e-4            # Lambda for LSQ fit (Try reducing back from 1e-2)
EVAL_BATCH_SIZE_ALL_LAYERS=32
### ----------------------------------------------------------------------
export HF_TOKEN= # Ensure your HF token is set if needed for private models
# tell HF to use the scratch mount that already holds the blobs
export HF_HOME=/mnt/ss-decoding/hf_cache            # master switch
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Optional: for potential fragmentation
# MODEL_ID and DATA_FILE are passed as args, no need to export if scripts use args

set -euo pipefail
mkdir -p "$ACT_DIR" "$HEAD_DIR" "$LOG_DIR"

echo "== 1. dump hidden states =="
# # Consider commenting out if acts already exist and config matches
# $PY dump_all.py \
#    --model "$MODEL_ID" \
#    --data "$DATA_FILE" \
#    --n "$N_PROMPTS" --batch "$BATCH" --seq "$SEQ" \
#    --out "$ACT_DIR"

echo "== 2. fit LSQ heads (single pass, all layers) =="
# The new fit_lsq.py handles all layers in one go, using GPU for accumulation in batches.
# It will use one GPU (default cuda:0). Set --gpu_id if needed.
# Adjust --layers_per_batch based on your GPU memory (e.g., H100 80GB might handle 24-30 layers per batch).
# $PY fit_lsq.py \
#    --act_dir "$ACT_DIR" \
#    --out_dir "$HEAD_DIR" \
#    --lambda_ "$FIT_LAMBDA" \
#    --layers_per_batch 24 # Example, adjust based on GPU memory for fitting

echo "== 3. one-pass evaluation (Scout, multi-GPU) =="
# Ensure all GPUs are visible to this single process
# The script eval_all_layers.py will use device_map="auto"
# to shard the model across all GPUs made visible here.
# The patch suggests CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 explicitly.
# If GPUS variable is correctly detecting 8, we can use seq.
VISIBLE_GPUS=$(seq -s, 0 $((GPUS - 1)))
echo "Running eval_all_layers.py with CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS"

CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS \
$PY eval_all_layers.py \
    --model "$MODEL_ID" \
    --data  "$DATA_FILE" \
    --heads "$HEAD_DIR" \
    --layers "0:$NUM_LAYERS:$LAYER_STEP" \
    --batch_size "$EVAL_BATCH_SIZE_ALL_LAYERS" \
    --seq "$SEQ" \
    --rows "train[50000:55000]" \
    | tee "$LOG_DIR/eval_all.out"

echo "== Evaluation complete. Log: $LOG_DIR/eval_all.out =="
echo "== Sweep finished =="

