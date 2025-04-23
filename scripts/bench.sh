#!/usr/bin/env bash
# Bench harness for Self‑Spec Llama‑4 repo.
set -euo pipefail

usage () {
  cat <<EOF
Usage: $0 RUN_NAME PROMPT_TOK GEN_TOK [options] [-- <extra vLLM flags>]

Required:
  RUN_NAME              label stored in results/<run_name>_*.json
  PROMPT_TOK            prompt tokens per request
  GEN_TOK               generation tokens per request

Options:
  -p, --num-prompts N   number of prompts (default: 64)
  -w, --warmup N        warm‑up iterations (default: 2)
  --dataset NAME        forwarded to benchmark_engine (--dataset ...)
  --preset NAME         quick|tiny fill common params (overrides above)
  -h, --help            this message
Any arguments after '--' are passed verbatim to vLLM benchmark_engine.
EOF
  exit 1
}

# ---------- defaults ----------
NUM_PROMPTS=64
WARMUP=2
DATASET=""
PRESET=""
# ------------------------------

# ---------- arg‑parse ----------
[[ $# -lt 3 ]] && usage
RUN_NAME=$1; shift
P_TOK=$1;  shift
G_TOK=$1;  shift
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--num-prompts) NUM_PROMPTS=$2; shift 2;;
    -w|--warmup)      WARMUP=$2; shift 2;;
    --dataset)        DATASET=$2; shift 2;;
    --preset)         PRESET=$2; shift 2;;
    --help|-h)        usage;;
    --)               shift; break;;        # everything after -- goes to vLLM
    *) break;;                              # start of extra flags, break loop
  esac
done
EXTRA_FLAGS=("$@")
# --------------------------------

# ---------- presets -------------
case "${PRESET:-}" in
  quick) NUM_PROMPTS=16; WARMUP=1;;
  tiny)  NUM_PROMPTS=4;  WARMUP=0;;
  "") ;;           # no preset
  *)  echo "Unknown preset '$PRESET'"; exit 1;;
esac
# --------------------------------

MODEL_DIR=${MODEL_DIR:-./model}
RESULT_DIR=results; mkdir -p "$RESULT_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT_JSON="${RESULT_DIR}/${RUN_NAME}_${STAMP}.json"
LOG_FILE="${OUT_JSON%.json}.log"

echo "▶️  $RUN_NAME  | prompt=$P_TOK gen=$G_TOK np=$NUM_PROMPTS warmup=$WARMUP"
echo "⏱  capturing SM util…"

TMP_UTIL=$(mktemp)
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 1 >"$TMP_UTIL" &
UTIL_PID=$!

cleanup () { kill $UTIL_PID 2>/dev/null || true; rm -f "$TMP_UTIL"; }
trap cleanup EXIT

# ---------- run benchmark -------
python -m vllm.entrypoints.benchmark_engine \
  --model "$MODEL_DIR" \
  --tokenizer "$MODEL_DIR" \
  --dtype auto \
  --download-dir "$MODEL_DIR" \
  --num-prompts "$NUM_PROMPTS" \
  --prompt-len "$P_TOK" \
  --gen-len "$G_TOK" \
  --warmup "$WARMUP" \
  ${DATASET:+--dataset "$DATASET"} \
  "${EXTRA_FLAGS[@]}" 2>&1 | tee "$LOG_FILE"

# ---------- metrics -------------
# 1. throughput etc.
JSON_LINE=$(grep -Eo 'JSON: \{.*\}' "$LOG_FILE" | tail -1 | sed 's/^JSON: //')
[[ -z $JSON_LINE ]] && { echo "❌ missing throughput JSON"; exit 1; }
data=$(python - <<PY "$JSON_LINE"
import json,sys,datetime,os,socket
d=json.loads(sys.argv[1])
d.update({"timestamp":datetime.datetime.utcnow().isoformat()+"Z",
          "hostname":socket.gethostname(),
          "git_commit":os.popen("git rev-parse --short HEAD").read().strip()})
print(json.dumps(d))
PY
)

# 2. accept rate if present
ACC=$(grep -Eo 'accept[^:]*:[[:space:]]*[0-9.]+|accepted.*\/.*' "$LOG_FILE" | tail -1 | \
      grep -Eo '[0-9.]+')
[[ -n ${ACC:-} ]] && data=$(python - <<PY "$data" "$ACC"
import json,sys,decimal
d=json.loads(sys.argv[1]); d["accept_rate"]=float(sys.argv[2]); print(json.dumps(d))
PY
)

# 3. SM util
SM_AVG=$(awk '{sum+=$1} END{ if(NR) printf "%.1f",sum/NR; }' "$TMP_UTIL")
data=$(python - <<PY "$data" "$SM_AVG" "$RUN_NAME" "$P_TOK" "$G_TOK"
import json,sys,os
d=json.loads(sys.argv[1])
d.update({"sm_util":float(sys.argv[2]),
          "run_name":sys.argv[3],
          "prompt_tok":int(sys.argv[4]),
          "gen_tok":int(sys.argv[5])})
print(json.dumps(d, indent=2))
PY
)

echo "$data" > "$OUT_JSON"
echo "📈  tokens/s: $(jq .tokens_per_second "$OUT_JSON") | "\
     "accept: $(jq -r '.accept_rate //"-"' "$OUT_JSON") | "\
     "SM util: $(jq .sm_util "$OUT_JSON")%  →  $OUT_JSON"