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

## Open items / decisions
- **ζ per job (user decision):** r50-sgd (QD-minus-SetB) for **Job B (OW)**; a separate
  all-intersecting-classes ζ for **Job A (CW)**. User to place weights; standalone
  `r50-sgd/best.pth` is not currently on disk (reported).
- **Eval protocol:** Sketch-DETR scores under the thesis §4 seed-14 binary-GT protocol (same as
  CASF), so baseline and method are one eval. See DEVIATIONS.md D1.

See DEVIATIONS.md for the full list of deviations from the paper spec.
