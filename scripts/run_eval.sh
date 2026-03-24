#!/bin/bash
# ============================================================
# AIMO3: Download checkpoint from Kaggle + run eval
# ============================================================
# Usage:
#   chmod +x run_eval.sh
#   ./run_eval.sh                        # defaults: 50 problems, pass@1
#   ./run_eval.sh --n_problems 100       # 100 problems
#   ./run_eval.sh --n_samples 8          # majority@8
#   ./run_eval.sh --step 500             # specific step checkpoint
#   ./run_eval.sh --n_problems 100 --n_samples 8 --step 1000

set -e  # exit on error

# ============================================================
# DEFAULTS
# ============================================================
N_PROBLEMS=50
N_SAMPLES=1
TEMPERATURE=0.7
MAX_TOKENS=2048
STEP="latest"
DATA_DIR="./data/nemotron-math-v2"
OUTPUT_DIR="./eval_results"

# load env
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

KAGGLE_USERNAME=${KAGGLE_USERNAME:-""}
if [ -z "$KAGGLE_USERNAME" ]; then
    echo "❌ KAGGLE_USERNAME not set in .env"
    exit 1
fi

# ============================================================
# PARSE ARGS
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_problems) N_PROBLEMS="$2"; shift 2 ;;
        --n_samples)  N_SAMPLES="$2";  shift 2 ;;
        --temperature)TEMPERATURE="$2";shift 2 ;;
        --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
        --step)       STEP="$2";       shift 2 ;;
        --data_dir)   DATA_DIR="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ============================================================
# SETUP
# ============================================================
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="$OUTPUT_DIR/eval_step${STEP}_p${N_PROBLEMS}_s${N_SAMPLES}_${TIMESTAMP}.json"
MODEL_DIR="./downloaded_model"

echo ""
echo "============================================================"
echo "  AIMO3 Eval"
echo "============================================================"
echo "  Kaggle user:  $KAGGLE_USERNAME"
echo "  Step:         $STEP"
echo "  Problems:     $N_PROBLEMS"
echo "  Samples:      $N_SAMPLES"
echo "  Temperature:  $TEMPERATURE"
echo "  Output:       $OUTPUT_FILE"
echo "============================================================"
echo ""

# ============================================================
# DOWNLOAD MODEL FROM KAGGLE
# ============================================================
echo "📥 Downloading merged model from Kaggle..."

HANDLE="${KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft-merged/transformers/default"

if [ -d "$MODEL_DIR" ]; then
    echo "  Found existing $MODEL_DIR — skipping download"
    echo "  (delete $MODEL_DIR to force re-download)"
else
    mkdir -p "$MODEL_DIR"

    # try kagglehub first
    python3 - <<EOF
import kagglehub, shutil, os
handle = "${HANDLE}"
print(f"  Downloading: {handle}")
path = kagglehub.model_download(handle)
print(f"  Downloaded to: {path}")
# copy to our target dir
if path != "${MODEL_DIR}":
    shutil.copytree(path, "${MODEL_DIR}", dirs_exist_ok=True)
    print(f"  Copied to: ${MODEL_DIR}")
EOF

    if [ $? -ne 0 ]; then
        echo "  kagglehub failed, trying kaggle CLI..."
        kaggle models instances versions download \
            "$HANDLE" \
            --path "$MODEL_DIR" \
            --untar
    fi
fi

echo "  ✅ Model ready at $MODEL_DIR"
echo ""

# ============================================================
# CHECK DATA DIR
# ============================================================
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data dir not found: $DATA_DIR"
    echo "   Set --data_dir or update DATA_DIR in script"
    exit 1
fi
echo "✅ Data dir found: $DATA_DIR"
echo ""

# ============================================================
# CHECK GPU
# ============================================================
echo "🖥️  GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader,nounits | \
    awk -F',' '{printf "   %s | %s/%s MB | util %s%%\n", $1, $2, $3, $4}'
echo ""

# ============================================================
# RUN EVAL
# ============================================================
echo "🚀 Running eval..."
echo ""

python3 eval_sft.py \
    --model       "$MODEL_DIR" \
    --data_dir    "$DATA_DIR" \
    --n_problems  "$N_PROBLEMS" \
    --n_samples   "$N_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --max_tokens  "$MAX_TOKENS" \
    --output      "$OUTPUT_FILE"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Eval complete!"
    echo "   Results saved: $OUTPUT_FILE"

    # print quick summary from json
    python3 - <<EOF
import json
with open("${OUTPUT_FILE}") as f:
    r = json.load(f)
print(f"\n  {'='*40}")
print(f"  Model:    ${STEP}")
print(f"  Correct:  {r['correct']}/{r['n_problems']}")
print(f"  Accuracy: {r['accuracy']:.1%}")
print(f"  Samples:  {r['n_samples']} per problem")
print(f"  {'='*40}\n")
EOF

else
    echo "❌ Eval failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

# ============================================================
# OPTIONAL: CLEANUP MODEL DIR TO SAVE DISK
# ============================================================
read -p "🗑  Delete downloaded model to free disk? (y/N): " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    rm -rf "$MODEL_DIR"
    echo "  Deleted $MODEL_DIR"
fi

echo ""
echo "Done!"
