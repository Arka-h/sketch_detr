# Sketch-DETR baseline — devlog

Reimplementation of Riba et al. 2021 ("Localizing ∞-shaped fishes", arXiv:2109.11874) as a
baseline for the CASF thesis. Built on vanilla facebookresearch/detr. Two jobs: (A) closed-world
calibration against the paper's published numbers (±0.010 mAP gate), (B) open-world on Set B.

All metrics reported as **decimals** (e.g. 0.419), never percentage points.

## Verified facts / gates passed
- **DETR reproduction:** unmodified DETR-R50 on COCO val2017 → mAP **0.419** / AP50 **0.623**
  vs paper 0.420 / 0.624. Eval pipeline, COCO path, pycocotools, checkpoint all confirmed.
- **Dataset:** `i%4==0` holdout reproduces **Set B exactly** (backpack, bicycle, clock, couch,
  dog, elephant, knife, mouse, oven, pizza, sandwich, skateboard, stop sign, train). seed-14
  category map is bit-deterministic; binary GT (category_id=0) builds.
- **Model:** both conditioning variants build; COCO-DETR init clean (unexpected=0); freezing =
  {image backbone ψ, sketch backbone ζ, transformer encoder}; 65.7M total / 10.8M trainable.
- **GT-calibration gate:** feed GT as predictions → **mAP 1.0000 / AP50 1.0000** (binary-GT,
  coordinate space, category-id wiring all correct).

## Wiring gates — all green (pipeline fully validated)
- **GT-calibration:** GT-as-prediction → mAP 1.0000 / AP50 1.0000.
- **Overfit (8 imgs):** loss 26.9→1.55, class_error 100→0.0 (escapes all-background basin ~it250);
  note: class_error legitimately sits at 100 for the first ~200 steps before flipping.
- **Determinism:** two `--eval` launches bit-identical 12-stat.
- **Driver:** train_one_epoch + per-epoch eval + checkpointing run end-to-end; 10.8M trainable;
  4.7 GB at bs=4 (so bs=16 fits easily).

## ζ plan (user decisions)
- **Job B (OW):** `r50-sgd` (QD-minus-Set B, 331-class, val_acc 0.8507) → `checkpoints/zeta_ow_r50sgd.pth`.
  Wired & verified. The ζ loader is format-agnostic (raw model_state | backbone_state | model |
  state_dict, ±module. prefix), asserts all 318 backbone tensors matched.
- **Job A (CW):** **56 COCO-intersecting** classes, paper-faithful (decision B) — **user will provide
  the trained RN50** (decision C); we do NOT train it. Drop at e.g. `checkpoints/zeta_cw56.pth`;
  `--sketch_ckpt` loads it directly.
- Known confound: OW ζ is 331-class, CW ζ is 56-class (asymmetric); see DEVIATIONS.md D2.

## OW data convention aligned to clip_ddetr_clean_run (2026-07-03)
- **Leak-free OW exclusion ported.** `coco_sketch.py` now wires `collect_image_ids` +
  `finalize_train_ids` (already present in `subset_select.py`) instead of the manual
  `if cname in visible` + `select_nested_subset_ids` path, which skipped exclusion entirely at
  `data_frac=1.0`. A training image containing ANY held-out (Set B) instance is now excluded in
  full at every fraction. Verified on real COCO train: OW-train 58,275 imgs, **0 overlap** with
  the 40,698 Set-B-containing images; closed-world unchanged at 98,973.
- Bumped `SUBSET_LOGIC_VERSION → v2-...-wired` (the old `v1` name claimed `fullUnseenExcl` but the
  wiring didn't honor it, so on-disk `qd_open_frac*` caches were mislabeled leaky) and deleted
  those stale caches.
- **Determinism aligned to clean_run:** eval is now ALWAYS deterministic (`args.eval or
  args.deterministic`), matching clean_run's `args.eval or args.deterministic_eval`.

## Ready to launch (scripts/RUNBOOK.md + scripts/run_job.sh)
- Job B (OW) is launch-ready now (leak-free). Job A fires the moment the CW RN50 lands.
- **Eval protocol:** scores under the thesis §4 seed-14 binary-GT protocol (same as CASF). D1.

See DEVIATIONS.md for the full list of deviations from the paper spec.
