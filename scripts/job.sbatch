#!/bin/bash
#SBATCH --job-name=sketch_detr
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@120          # warn 120s before time limit so we requeue cleanly
#SBATCH --requeue
#
# Cluster launcher for a Sketch-DETR cell. Auto-resumes from $OUT/checkpoint.pth, so a
# requeue/preemption continues seamlessly. Fill the env for your cluster, then:
#   sbatch --export=ALL,WORLD=closed,DATASET=qd,COND=encoder_concat,\
#     CKPT=outputs/rn50_qd_cw56/best.pth,OUT=outputs/jobA_cw_qd_encconcat scripts/job.sbatch
#
# Adjust: partition/account, module loads, conda env, COCO_PATH, SKETCH_HOME, WANDB_MODE.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- environment (EDIT for your cluster) ---
# module load cuda/12.x
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate locformer
export PYTHON="${PYTHON:-/home/rahul/miniconda3/envs/locformer/bin/python}"
export COCO_PATH="${COCO_PATH:-/mnt/1tb/data/coco}"
export SKETCH_HOME="${SKETCH_HOME:-/mnt/1tb/data}"
export WANDB="${WANDB:-1}"                 # log to wandb (project sketch_detr)
export WANDB_MODE="${WANDB_MODE:-offline}" # offline on air-gapped clusters; sync later

# requeue this job on USR1 (sent 120s before timeout); auto-resume continues from checkpoint
trap 'echo "[sbatch] caught USR1 -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1

srun bash scripts/run_job.sh
