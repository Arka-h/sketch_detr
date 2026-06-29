# Sketch-DETR reimplementation — deviations from the paper spec

Deliverable #4 (handover §6): every deviation from Riba et al.'s spec, with reason.

## D1 — Eval scoring protocol (binary seed-14 GT)
- **Spec / thesis code:** `ow_repr/engine.evaluate` scores with the standard `CocoEvaluator`
  over the full held-out GT.
- **What we do:** handover §4 binary seed-14 GT — one category per val image (deterministic
  `random.Random(14)` over the sorted present categories), GT rebuilt with `category_id=0`,
  pycocotools full 12-stat. Predictions are class-agnostic (foreground only).
- **Why:** the handover mandates §4 for all reported cells; it is the leak-free per-image
  protocol and avoids cross-category contamination when an image has >1 held-out category.

## D2 — Sketch backbone ζ (trained in-house; two checkpoints)
- **Spec:** ζ = ResNet-50 sketch classifier trained on the COCO-intersecting sketch classes.
  Closed-world → all 56; open-world → seen 42.
- **What we do:** train ζ ourselves (ImageNet-init ResNet-50, fc→#classes, finetuned QD classifier;
  f_s = 2048-d global-pooled feature), **two checkpoints**:
    - `zeta_qd_cw56_resnet50.pth` — all 56 COCO∩QD classes → **Job A (CW)**.
    - `zeta_qd_ow42_resnet50.pth` — 42 seen (exclude Set B) → **Job B (OW)**.
- **Why not reuse LocFormer's ζ:** (1) **CW fairness** — LocFormer's ζ excludes Set B, which would
  deflate the CW reproduction on those 14 categories and bias us against the ±0.010 gate; the paper's
  CW ζ sees all intersecting classes. (2) **OW contamination risk** — the only available LocFormer ζ
  was extracted from the `lf_qd_rn50` *detector*, which does **not** freeze `sketch_embedding`, so ζ
  was finetuned during detector training and may have seen Set B sketches as queries. Training fresh
  removes both issues and makes ζ a pure classifier (closer to the paper).
- **Recipe deviation:** the paper trains ζ on the COCO-intersecting subset (56/42); LocFormer trained
  on ~331 QD classes minus Set B. We follow the paper (56/42).

## D3 — Sketch normalisation
- **Spec:** unspecified preprocessing for ζ.
- **What we do:** ImageNet mean/std (ζ is ImageNet-init), sketches rendered white-on-black
  (QD stroke-3 rasterised; Sketchy PNGs inverted). NOT CLIP normalisation (that is CASF-specific).

## D4 — Single sketch query (k=1)
- **Spec / handover §4:** single sketch per query. `ow_repr` renders k=3; `clean_run` uses k=1.
- **What we do:** k=1 (deterministic single sketch at val via `Random(14)`), matching §4.
