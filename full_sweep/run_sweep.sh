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
BATCH=256                  # GPU batch during dump (Increased from 8 to 256)
LAYER_STEP=4               # evaluate every 4th layer
GPUS=$(nvidia-smi -L | wc -l)   # auto-detect
NUM_LAYERS=48              # Set to the total number of layers in your model
ACT_DIR="acts"
HEAD_DIR="heads"
LOG_DIR="logs"
FIT_LAMBDA=1e-4            # Lambda for LSQ fit (Try reducing back from 1e-2)
### ----------------------------------------------------------------------
export HF_TOKEN=
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
$PY fit_lsq.py \
   --act_dir "$ACT_DIR" \
   --out_dir "$HEAD_DIR" \
   --lambda_ "$FIT_LAMBDA" \
   --gpu_id 0 \
   --layers_per_batch 16 \
   2>&1 | tee "$LOG_DIR/fit_lsq_all_layers.log"
# If the above tee command causes issues with tqdm rendering due to piping,
# an alternative is to let tqdm print to stderr if fit_lsq.py is modified,
# or simply tail the log file in another terminal: tail -f logs/fit_lsq_all_layers.log
echo "All LSQ head fitting complete. See $LOG_DIR/fit_lsq_all_layers.log"

echo "== 3. parallel evaluation =="
MAX_LAYER_IDX=$((NUM_LAYERS - 1)) 
LAYERS_TO_EVAL=($(seq 0 $LAYER_STEP $MAX_LAYER_IDX)) 
CURRENT_GPU_EVAL=0
JOB_COUNT_EVAL=0
for L_EVAL in "${LAYERS_TO_EVAL[@]}"; do
  HEAD_FILE_PATH="$HEAD_DIR/h$L_EVAL.pt"
  if [ ! -f "$HEAD_FILE_PATH" ]; then
    echo "Warning: Head file $HEAD_FILE_PATH not found. Skipping evaluation for layer $L_EVAL."
    continue
  fi
  echo "Starting evaluation for layer $L_EVAL on GPU $CURRENT_GPU_EVAL"
  CUDA_VISIBLE_DEVICES=$CURRENT_GPU_EVAL \
    $PY eval_one.py \
        --model "$MODEL_ID" \
        --data "$DATA_FILE" \
        --layer "$L_EVAL" \
        --head "$HEAD_FILE_PATH" \
        > "$LOG_DIR/l$L_EVAL.out" 2>&1 &

  CURRENT_GPU_EVAL=$(( (CURRENT_GPU_EVAL + 1) % GPUS ))
  JOB_COUNT_EVAL=$((JOB_COUNT_EVAL + 1))

  if (( JOB_COUNT_EVAL >= GPUS )); then
    wait -n # Wait for any one background job to finish
    JOB_COUNT_EVAL=$((JOB_COUNT_EVAL - 1))
  fi
done
wait # Wait for all remaining background evaluation jobs
echo "all done – see $LOG_DIR for per-layer accept-rates"

