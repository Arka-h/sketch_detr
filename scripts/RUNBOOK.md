# Sketch-DETR runbook — Job A (CW calibration) & Job B (OW)

Everything is wired and gated (GT-calib mAP 1.0, overfit class_err→0, determinism bit-identical).
The only missing input is the **CW ζ** (closed-world sketch classifier). Job B's OW ζ is ready.

## ζ checkpoints

| ζ | classes | file | status |
|---|---|---|---|
| OW ζ | 331 (QD − Set B) | `checkpoints/zeta_ow_r50sgd.pth` (= your `r50-sgd/best.pth`) | ✅ wired, val_acc 0.8507 |
| **CW ζ** | **56 COCO∩QD** (paper-faithful) | **you drop it** → e.g. `checkpoints/zeta_cw56.pth` | ⏳ awaiting |

The loader is format-agnostic: drop the CW RN50 in **any** form (raw `r50-sgd`-style
`{model_state, val_acc, epoch}`, or `model`/`state_dict`/`backbone_state`, with or without
`module.` prefix). `--sketch_ckpt <path>` loads it directly — no conversion. It hard-asserts
all 318 ResNet-50 backbone tensors matched, so a wrong checkpoint fails loudly.

## When the CW ζ drops — Job A (do FIRST; the ±0.010 gate)

```bash
# PRIMARY: closed-world QD, Encoder-Concat  → target paper mAP 0.414 / AP50 0.621
WORLD=closed DATASET=qd COND=encoder_concat CKPT=checkpoints/zeta_cw56.pth \
  OUT=outputs/jobA_cw_qd_encconcat bash scripts/run_job.sh

# PRIMARY: closed-world Sketchy, Encoder-Concat → target paper mAP 0.420 / AP50 0.636
WORLD=closed DATASET=sketchy COND=encoder_concat CKPT=checkpoints/zeta_cw56.pth \
  OUT=outputs/jobA_cw_sk_encconcat bash scripts/run_job.sh

# SECONDARY: Object-Query variant (paper QD 0.387 / SK 0.333)
WORLD=closed DATASET=qd      COND=object_query CKPT=checkpoints/zeta_cw56.pth OUT=outputs/jobA_cw_qd_objquery bash scripts/run_job.sh
WORLD=closed DATASET=sketchy COND=object_query CKPT=checkpoints/zeta_cw56.pth OUT=outputs/jobA_cw_sk_objquery bash scripts/run_job.sh
```

**Gate:** Encoder-Concat reproduced mAP within **±0.010** of QD 0.414 / SK 0.420. If missed → STOP,
report; do NOT run Job B. (mAP reported in decimals.)

## After Job A passes — Job B (OW on Set B)

```bash
WORLD=open DATASET=qd COND=encoder_concat CKPT=checkpoints/zeta_ow_r50sgd.pth \
  OUT=outputs/jobB_ow_qd_encconcat bash scripts/run_job.sh
# Object-Query optional; Sketchy OW optional.
```

## What to watch (per-epoch `[epoch N] mAP=... ` lines in `train.log`)

- `class_error` sits at 100 for the first ~200 steps then flips — **expected**, not a bug.
- Tripwires: `AP_s < AP_l` (small worst), `class_error` not stuck at 100 past epoch 1.
- Job B sanity: OW mAP ≪ CW mAP. If OW ≈ CW → suspect a leak, re-check the partition.
- Each run writes `provenance.txt` (commit, seed, Set B list) + `log.txt` (per-epoch 12-stat).

## Eval-only (any saved checkpoint)

```bash
SKETCH_HOME=/mnt/1tb/data python main_sketch.py --eval --deterministic \
  --sketch_cond encoder_concat --sketch_ckpt <ζ> --resume <run>/checkpoint.pth \
  --train_scheme_world <closed|open> --sketch_dataset <qd|sketchy> \
  --coco_path /mnt/1tb/data/coco --batch_size 4
```
