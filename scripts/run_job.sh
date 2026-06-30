#!/bin/bash
# Parametrized Sketch-DETR launcher (Job A closed-world / Job B open-world).
#
# Usage:
#   WORLD=closed|open  DATASET=qd|sketchy  COND=encoder_concat|object_query \
#   CKPT=path/to/zeta.pth  OUT=outputs/run_name  [EPOCHS=50] [BS=4]  bash scripts/run_job.sh
#
# Examples:
#   # Job A — closed-world QD, Encoder-Concat (PRIMARY), against paper 0.414/0.621
#   WORLD=closed DATASET=qd COND=encoder_concat CKPT=checkpoints/zeta_cw56.pth \
#     OUT=outputs/jobA_cw_qd_encconcat bash scripts/run_job.sh
#   # Job B — open-world QD on Set B, Encoder-Concat (PRIMARY)
#   WORLD=open DATASET=qd COND=encoder_concat CKPT=checkpoints/zeta_ow_r50sgd.pth \
#     OUT=outputs/jobB_ow_qd_encconcat bash scripts/run_job.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${WORLD:?set WORLD=closed|open}"
: "${DATASET:?set DATASET=qd|sketchy}"
: "${COND:?set COND=encoder_concat|object_query}"
: "${CKPT:?set CKPT=path/to/zeta.pth}"
: "${OUT:?set OUT=outputs/run_name}"
EPOCHS="${EPOCHS:-50}"
BS="${BS:-16}"                 # ~18GB at bs=16 (4.7GB at bs=4); RTX 8000 has 48GB
EVAL_EVERY="${EVAL_EVERY:-5}"  # full-val eval cadence (final epoch always evaluated)
WORKERS="${WORKERS:-8}"
COCO_PATH="${COCO_PATH:-/mnt/1tb/data/coco}"
PYTHON="${PYTHON:-/home/rahul/miniconda3/envs/locformer/bin/python}"
export SKETCH_HOME="${SKETCH_HOME:-/mnt/1tb/data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

[ -f "$CKPT" ] || { echo "ERROR: ζ checkpoint not found: $CKPT"; exit 1; }
mkdir -p "$OUT"

echo "[run_job] world=$WORLD dataset=$DATASET cond=$COND ckpt=$CKPT out=$OUT epochs=$EPOCHS bs=$BS"
echo "[run_job] provenance: $(git rev-parse --short HEAD 2>/dev/null) | seed=42 | $(date -Is)" | tee "$OUT/provenance.txt"
echo "[run_job] holdout(SetB if open): backpack bicycle clock couch dog elephant knife mouse oven pizza sandwich skateboard 'stop sign' train" | tee -a "$OUT/provenance.txt"

"$PYTHON" -u main_sketch.py \
    --sketch_cond "$COND" \
    --sketch_ckpt "$CKPT" \
    --detr_init checkpoints/detr-r50-e632da11.pth \
    --train_scheme_world "$WORLD" \
    --sketch_dataset "$DATASET" \
    --num_sketches 1 \
    --dataset_file coco_sketch \
    --coco_path "$COCO_PATH" \
    --epochs "$EPOCHS" --lr_drop 40 \
    --lr 1e-4 --weight_decay 1e-4 --clip_max_norm 0.1 \
    --batch_size "$BS" --num_workers "$WORKERS" \
    --eval_every "$EVAL_EVERY" \
    ${WANDB:+--wandb --wandb_mode "${WANDB_MODE:-online}"} \
    --output_dir "$OUT" 2>&1 | tee -a "$OUT/train.log"
# wandb: set WANDB=1 (project 'sketch_detr', entity aurkohaldi). WANDB_MODE=offline on an
# offline cluster (sync later with `wandb sync wandb/offline-run-*`). Auto-resumes from
# $OUT/checkpoint.pth on relaunch (cluster requeue-safe). tee -a so resumes append.
# NOTE: training runs fast (AMP, seeded, non-strict kernels). The OFFICIAL reported
# number is a SEPARATE deterministic eval pass on the best checkpoint:
#   python main_sketch.py --eval --deterministic --sketch_cond <C> --sketch_ckpt <ζ> \
#     --resume "$OUT/checkpoint.pth" --train_scheme_world "$WORLD" --sketch_dataset "$DATASET" \
#     --coco_path "$COCO_PATH" --batch_size 4
