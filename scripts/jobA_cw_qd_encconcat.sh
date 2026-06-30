#!/bin/bash
#SBATCH --job-name=jobA_cw_qd_encconcat          # Job name
#SBATCH --output=outputs/jobA_cw_qd_encconcat_%j.log   # Standard output log (%j = job ID)
#SBATCH --error=outputs/jobA_cw_qd_encconcat_%j.err    # Standard error log
#SBATCH --time=2-00:00:00                     # Time limit (dd-hh:mm:ss)
#SBATCH --ntasks=2                            # Number of tasks
#SBATCH --cpus-per-task=10                    # Number of CPUs per task
#SBATCH --mem=60GB                            # Memory allocation
#SBATCH --partition=ada                       # Partition (long/queue)
#SBATCH --gres=gpu:ADA6000:2                  # GPU allocation (2x ADA6000)
#SBATCH --account=research
# #SBATCH --nodelist=cn8                        # Node to run on (modify as needed)
# =============================================================
# Sketch-DETR — Job A reproduction: CLOSED-WORLD, QuickDraw, Encoder-Concat (PRIMARY).
# Calibrates against the paper's published closed-world numbers (mAP 0.414 / AP50 0.621).
# Style mirrors clip_ddetr scripts/floor_cw.sh: torch.distributed.run sets RANK/WORLD_SIZE
# (env:// init), so init_distributed_mode never reads SLURM_NTASKS. TOTAL batch is held at
# 16 (= the local repro run) by splitting per-GPU, so multi-GPU doesn't drift the recipe.

echo "job: $SLURM_JOB_NAME"
# >>> Conda setup <<<
source ~/miniconda3/etc/profile.d/conda.sh
conda activate locformer

# Job execution commands
. ./.env
echo $COCO_HOME
echo $SLURM_JOBID

# 1) Find a free port by binding to port 0
export MASTER_PORT=$(python - <<'EOF'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 0))
port = s.getsockname()[1]
s.close()
print(port)
EOF
)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_NNODES=${SLURM_NNODES:-1}
export SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-1}
echo "nnodes: $SLURM_NNODES"
echo "nproc_per_node: $SLURM_GPUS_ON_NODE"
echo "master port: $MASTER_PORT"

# Auto-resume from the last checkpoint if present (saved BEFORE eval each epoch).
# (main_sketch also auto-resumes from $OUT_DIR/checkpoint.pth; --resume here makes it explicit
# and requeue-safe — same wandb run continues via the stored run id.)
OUT_DIR="$PROJECT_HOME/outputs/jobA_cw_qd_encconcat"
RESUME_ARG=""
if [ -f "$OUT_DIR/checkpoint.pth" ]; then
    RESUME_ARG="--resume $OUT_DIR/checkpoint.pth"
    echo "Resuming from $OUT_DIR/checkpoint.pth"
else
    echo "No checkpoint found; starting fresh."
fi

# Keep TOTAL batch = 16 regardless of GPU count (per-GPU = 16 / nproc), so the recipe matches
# the single-GPU repro run exactly.
TOTAL_BS=${TOTAL_BS:-16}
PER_GPU_BS=$(( TOTAL_BS / SLURM_GPUS_ON_NODE ))
[ "$PER_GPU_BS" -lt 1 ] && PER_GPU_BS=1
echo "total_bs: $TOTAL_BS  per_gpu_bs: $PER_GPU_BS"

python -u -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --master_port $MASTER_PORT \
    main_sketch.py \
    --output_dir $OUT_DIR \
    --coco_path $COCO_HOME \
    --dataset_file coco_sketch \
    --sketch_cond encoder_concat \
    --sketch_ckpt outputs/rn50_qd_cw56/best.pth \
    --detr_init checkpoints/detr-r50-e632da11.pth \
    --train_scheme_world closed \
    --sketch_dataset qd \
    --num_sketches 1 \
    --epochs 50 --lr_drop 40 \
    --lr 1e-4 --weight_decay 1e-4 --clip_max_norm 0.1 \
    --batch_size $PER_GPU_BS --num_workers 8 \
    --eval_every 5 \
    --wandb --wandb_mode ${WANDB_MODE:-online} \
    --wandb_name jobA_cw_qd_encconcat_${SLURM_JOBID:-local} \
    $RESUME_ARG
