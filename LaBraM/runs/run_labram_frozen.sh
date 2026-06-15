#!/bin/bash
# LaBraM frozen-backbone fine-tuning (linear probe: train task head only).
#
# Usage:
#   TASK=age SEGMENT=1s bash runs/run_labram_frozen.sh
#   TASK=gender SEGMENT=2s bash runs/run_labram_frozen.sh --output_dir /path/to/output --log_dir /path/to/logs
#   TASK=regression SEGMENT=4s bash runs/run_labram_frozen.sh
#
# TASK: age | gender | regression
# SEGMENT: 1s | 2s | 4s

set -eo pipefail

TASK="${TASK:?Set TASK to age, gender, or regression}"
SEGMENT="${SEGMENT:?Set SEGMENT to 1s, 2s, or 4s}"

# Conda/Qt activate.d scripts reference optional env vars (e.g. QT_XCB_GL_INTEGRATION);
# nounset (-u) must be off during conda activate (same pattern as other LaBraM run scripts).
set +u
if [ -f ~/.bashrc ]; then
    source ~/.bashrc 2>/dev/null || true
fi
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
fi
if command -v conda &> /dev/null; then
    if conda env list 2>/dev/null | grep -q "eeg_env"; then
        conda activate eeg_env
    else
        conda activate base 2>/dev/null || true
    fi
fi

if command -v module &> /dev/null; then
    module load python/3.10 2>/dev/null || true
    module load cuda/12.6 2>/dev/null || true
    module load cudnn/8.9.7 2>/dev/null || \
    module load cudnn/9.0 2>/dev/null || \
    module load cudnn 2>/dev/null || true
fi
set -u

CONDA_PYTHON="$HOME/miniconda3/envs/eeg_env/bin/python"
if [ -f "$CONDA_PYTHON" ]; then
    export PATH="$HOME/miniconda3/envs/eeg_env/bin:$PATH"
fi
PYTHON_CMD="${PYTHON_PATH:-python}"
if [ -f "$CONDA_PYTHON" ]; then
    PYTHON_CMD="$CONDA_PYTHON"
fi

unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT SLURM_PROCID

OUTPUT_DIR_OVERRIDE=""
LOG_DIR_OVERRIDE=""
OTHER_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --output_dir) OUTPUT_DIR_OVERRIDE="$2"; shift 2 ;;
    --log_dir)    LOG_DIR_OVERRIDE="$2"; shift 2 ;;
    *)            OTHER_ARGS+=("$1"); shift ;;
  esac
done
set -- "${OTHER_ARGS[@]}"

case "$TASK" in
  age)
    DATASET="age_baseline"
    NB_CLASSES=3
    RUN_LABEL="labram_age_frozen"
    ;;
  gender)
    DATASET="gender_baseline"
    NB_CLASSES=1
    RUN_LABEL="labram_gender_frozen"
    ;;
  regression)
    DATASET="age_regression_baseline"
    NB_CLASSES=1
    RUN_LABEL="labram_age_regression_frozen"
    ;;
  *)
    echo "Unknown TASK=$TASK (expected age, gender, or regression)" >&2
    exit 1
    ;;
esac

case "$SEGMENT" in
  1s) BATCH_SIZE=1024 ;;
  2s) BATCH_SIZE=512 ;;
  4s) BATCH_SIZE=256 ;;
  *)
    echo "Unknown SEGMENT=$SEGMENT (expected 1s, 2s, or 4s)" >&2
    exit 1
    ;;
esac

EPOCHS=50
LEARNING_RATE=0.0005
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=5
MIN_LR=1e-5
DROP_PATH=0.1
SMOOTHING=0.0
LAYER_DECAY=0.65
CLIP_GRAD=3.0
LAYER_SCALE_INIT_VALUE=0.1
MODEL_EMA_DECAY=0.996

MODEL="labram_base_patch200_200"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EEG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PRETRAINED_PATH="${PRETRAINED_PATH:-$EEG_ROOT/LaBraM/checkpoints/labram-base.pth}"
DATA_PATH="${DATA_PATH:-/home/mojtabam/scratch/processed_eeg_data_hdf5}"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${DATASET}_${SEGMENT}_frozen}"
LOG_DIR="${LOG_DIR:-./logs/${DATASET}_${SEGMENT}_frozen}"
[[ -n "${OUTPUT_DIR_OVERRIDE:-}" ]] && OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
[[ -n "${LOG_DIR_OVERRIDE:-}" ]] && LOG_DIR="$LOG_DIR_OVERRIDE"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "=========================================="
echo "LaBraM frozen fine-tuning: ${RUN_LABEL}_${SEGMENT}"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Segment: $SEGMENT"
echo "Batch size: $BATCH_SIZE"
echo "Output: $OUTPUT_DIR"
echo "Pretrained: $PRETRAINED_PATH"
echo "=========================================="

cd "$SCRIPT_DIR/.."
export PYTHONUNBUFFERED=1

"$PYTHON_CMD" run_class_finetuning.py \
    --model "$MODEL" \
    --finetune "$PRETRAINED_PATH" \
    --dataset "$DATASET" \
    --nb_classes "$NB_CLASSES" \
    --data_path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --min_lr "$MIN_LR" \
    --drop_path "$DROP_PATH" \
    --smoothing "$SMOOTHING" \
    --segment_length "$SEGMENT" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --abs_pos_emb \
    --qkv_bias \
    --use_mean_pooling \
    --layer_decay "$LAYER_DECAY" \
    --layer_scale_init_value "$LAYER_SCALE_INIT_VALUE" \
    --opt_betas 0.9 0.98 \
    --opt_eps 1e-8 \
    --momentum 0.9 \
    --clip_grad "$CLIP_GRAD" \
    --save_ckpt \
    --auto_resume \
    --seed 42 \
    --pin_mem \
    --aggregate_by_participant majority_vote \
    --freeze_backbone \
    --unfreeze_last_n_blocks 0 \
    "$@"

echo "Fine-tuning completed: ${RUN_LABEL}_${SEGMENT}"
