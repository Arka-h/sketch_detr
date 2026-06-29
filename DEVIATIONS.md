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
- **What we do (final, per user):**
    - **Job B (OW):** `r50-sgd/best.pth` — the user's pretrained ResNet-50, **331 QD classes minus
      Set B**, val_acc 0.8507 → `checkpoints/zeta_ow_r50sgd.pth`. Leak-free (Set B never a label)
      and a rich general sketch encoder. A ζ exposed to *more non-held-out* classes is legitimately
      *stronger* (better embedding geometry → better transfer), not contamination — only Set B as a
      label would be a leak.
    - **Job A (CW):** **56 COCO-intersecting** classes, paper-faithful — **user provides the trained
      RN50** (we do not train it). `--sketch_ckpt` loads it directly (format-agnostic loader).
- **Earlier wrong turn (corrected):** I first *extracted* ζ from the `lf_qd_rn50` detector's
  `sketch_embedding.*`. That detector does **not** freeze the sketch encoder, so that copy was
  detector-finetuned (contamination-risky). Discarded in favour of the clean standalone `r50-sgd`.
- **Known confound:** OW ζ is 331-class, CW ζ is 56-class — asymmetric coverage between the two jobs.
  Reported so the CW↔OW gap isn't over-attributed to category transfer alone.

## D3 — Sketch normalisation
- **Spec:** unspecified preprocessing for ζ.
- **What we do:** ImageNet mean/std (ζ is ImageNet-init), sketches rendered white-on-black
  (QD stroke-3 rasterised; Sketchy PNGs inverted). NOT CLIP normalisation (that is CASF-specific).

## D5 — Mixed precision (AMP) training + train/eval determinism split
- **Spec:** unspecified; paper presumably fp32.
- **What we do:** train with AMP (autocast + GradScaler) for throughput on the RTX 8000.
  Training uses seeded-but-non-strict kernels (fast); the **official reported eval is a separate
  `--eval --deterministic` pass** (bit-identical, verified). Per-epoch evals during training are
  monitoring-only. Numerical impact of AMP on final mAP is negligible.

## D6 — Compute budget (RTX 8000, single GPU)
- 50-epoch closed-world QD ≈ 82 min/epoch (~2.8 days/cell). Recorded so any epoch-budget or
  early-stopping decision (plateau-based) is explicit, not silent. See DEVLOG for the chosen budget.

## D4 — Single sketch query (k=1)
- **Spec / handover §4:** single sketch per query. `ow_repr` renders k=3; `clean_run` uses k=1.
- **What we do:** k=1 (deterministic single sketch at val via `Random(14)`), matching §4.
