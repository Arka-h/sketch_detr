"""Seeded, nested, persisted data-fraction subset selection (ablations branch).

Single source of truth for the data-scaling subset, shared by the live dataset
(`datasets/coco.py`) and the offline builder (`scripts/build_subsets.py`) so the
id-list a training run draws is bit-identical to the one persisted offline.

Fixes the four documented defects of the old inline `set(np.random.choice(...))`:
  1. seeded + persisted    -> np.random.default_rng(subset_seed), id-list to disk
  2. without replacement    -> exact fraction (no set() collapse of duplicates)
  3. nested 0.25 ⊂ 0.50 ⊂ 1.00 -> permute the FULL per-class list once, slice a prefix
  4. sketch pool untouched  -> only the image set shrinks (handled in coco.py)
plus a fifth correctness fix this ablation exposed:
  5. the held-out (Set B / unseen) exclusion uses the FULL unseen set at every
     fraction. Subsetting unseen (the old code did) means <100% runs exclude fewer
     held-out images, leaking unseen-category images into training. At 100% the two
     are identical, which is why the headline was unaffected.

Everything here is torch-free on purpose (numpy + json + stdlib only).
"""

import json
import os
from typing import Dict, List, Set, Tuple

import numpy as np

# Bump this whenever the selection logic below changes in a way that alters the
# produced image set. Cached train-jsons stamped with a different version are
# treated as stale and regenerated — so a logic change can never be silently
# served from an old cache (the failure mode the unseeded draw created).
SUBSET_LOGIC_VERSION = "v1-seeded-nested-fullUnseenExcl"


def open_world_split(all_categories: List[str]) -> Tuple[List[str], List[str]]:
    """Reproduce coco.py's open-world holdout: every 4th category (i % 4 == 0) is
    unseen (Set B). Returns (seen_cats, unseen_cats)."""
    unseen = [all_categories[i] for i in range(len(all_categories)) if i % 4 == 0]
    seen = [all_categories[i] for i in range(len(all_categories)) if i % 4 != 0]
    return seen, unseen


def collect_image_ids(json_file: dict, id2class: Dict[int, str], all_categories: List[str],
                      unseen_cats: List[str], image_set: str, train_scheme_world: str
                      ) -> Tuple[list, Dict[int, list], Dict[int, list]]:
    """One pass over annotations, matching coco.py exactly. Returns
    (annotate, seen_image_ids, unseen_image_ids) where the *_image_ids are
    {category_id: [image_id, ...]} (with duplicates, as the original built them)."""
    annotate: list = []
    seen_image_ids: Dict[int, list] = {}
    unseen_image_ids: Dict[int, list] = {}
    all_set = set(all_categories)
    unseen_set = set(unseen_cats)
    for anno in json_file['annotations']:
        cat_id = anno['category_id']
        img_id = anno['image_id']
        cname = id2class[cat_id]
        if cname not in all_set:
            continue
        if image_set == 'train' or train_scheme_world == 'closed':
            annotate.append(anno)
            if cname not in unseen_set:
                seen_image_ids.setdefault(cat_id, []).append(img_id)
            else:
                unseen_image_ids.setdefault(cat_id, []).append(img_id)
        else:  # val (open): keep only the held-out categories as the eval pool
            if cname in unseen_set:
                annotate.append(anno)
                seen_image_ids.setdefault(cat_id, []).append(img_id)
    return annotate, seen_image_ids, unseen_image_ids


def select_nested_subset_ids(image_ids_by_cat: Dict[int, list], data_frac: float,
                             subset_seed: int) -> Dict[int, Set[int]]:
    """Per-category nested subset of unique image ids.

    For each category (iterated in sorted category-id order for a deterministic RNG
    consumption order), take its UNIQUE image ids in sorted order, permute them once
    with a seeded RNG, and keep the first ``int(data_frac * n_unique)``. Because the
    permutation is computed over the FULL list independently of ``data_frac`` and
    sliced as a prefix, selections nest: 0.25 ⊂ 0.50 ⊂ 1.00. Sampling is without
    replacement, so the fraction is exact. Returns {category_id: set(image_ids)}.
    """
    if not (0.0 < data_frac <= 1.0):
        raise ValueError(f"data_frac must be in (0, 1], got {data_frac}")
    rng = np.random.default_rng(subset_seed)
    out: Dict[int, Set[int]] = {}
    for cat_id in sorted(image_ids_by_cat):
        uniq = sorted(set(image_ids_by_cat[cat_id]))
        n = len(uniq)
        perm = rng.permutation(n)            # consumes the same RNG state for any data_frac
        k = int(data_frac * n)
        out[cat_id] = {uniq[i] for i in perm[:k]}
    return out


def finalize_train_ids(seen_image_ids: Dict[int, list], unseen_image_ids: Dict[int, list],
                       data_frac: float, subset_seed: int
                       ) -> Tuple[Set[int], Dict[int, int]]:
    """Final training image-id set for a given fraction.

    Subset SEEN only (seeded, nested); exclude the FULL unseen set (fix #5).
    Returns (final_image_ids, per_class_counts) where per_class_counts is the count
    of surviving images per seen category (post unseen-overlap removal) for the
    starvation eyeball.
    """
    subset_seen = select_nested_subset_ids(seen_image_ids, data_frac, subset_seed)
    seen: Set[int] = set().union(*subset_seen.values()) if subset_seen else set()
    unseen: Set[int] = set().union(*unseen_image_ids.values()) if unseen_image_ids else set()
    seen -= unseen  # held-out images never train (full unseen set, every fraction)
    per_class = {cat_id: len(ids - unseen) for cat_id, ids in subset_seen.items()}
    return seen, per_class


def subset_bundle_path(out_dir: str, sketch_name: str, world: str,
                       data_frac: float, subset_seed: int) -> str:
    return os.path.join(out_dir, f"{sketch_name}_{world}_frac{data_frac:.2f}_seed{subset_seed}.json")


def persist_subset_bundle(out_dir: str, sketch_name: str, world: str, data_frac: float,
                          subset_seed: int, image_ids: Set[int], per_class_counts: Dict[int, int],
                          id2class: Dict[int, str], git_commit: str = "", extra: dict = None) -> str:
    """Write-if-absent + verify-on-exists. The persisted id-list is the reproducibility
    artifact; if a later build draws a *different* set for the same (fraction, seed),
    raise loudly rather than silently overwrite — that mismatch is exactly the failure
    mode this project rebuilt to eliminate. Returns the bundle path."""
    os.makedirs(out_dir, exist_ok=True)
    path = subset_bundle_path(out_dir, sketch_name, world, data_frac, subset_seed)
    ids_sorted = sorted(int(i) for i in image_ids)
    payload = {
        "sketch_name": sketch_name,
        "world": world,
        "fraction": round(float(data_frac), 2),
        "subset_seed": int(subset_seed),
        "n_images": len(ids_sorted),
        "git_commit": git_commit,
        "per_class_counts": {id2class.get(int(k), str(k)): int(v)
                             for k, v in sorted(per_class_counts.items())},
        "image_ids": ids_sorted,
    }
    if extra:
        payload.update(extra)
    if os.path.exists(path):
        with open(path) as f:
            prev = json.load(f)
        if prev.get("image_ids") != ids_sorted:
            raise RuntimeError(
                f"[SUBSET MISMATCH] {path} already on disk with a DIFFERENT image set "
                f"(prev n={prev.get('n_images')} vs new n={len(ids_sorted)}). Refusing to "
                f"overwrite a persisted subset. Same (fraction, seed) must give the same ids.")
        return path
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Filtered-train-json cache (efficiency: skip the ~469MB scan + draw on reuse)
# ---------------------------------------------------------------------------
# The cache filename encodes sketch · world · fraction · seed (everything that
# changes the image set), and a tiny sidecar `.meta` records SUBSET_LOGIC_VERSION
# + the same key. A cache is reused only if the sidecar matches the current code
# version and request — so it is exactly the file a fresh run would have written,
# never a stale or differently-computed one.

def train_cache_path(out_dir: str, sketch_name: str, world: str,
                     data_frac: float, subset_seed: int) -> str:
    return os.path.join(out_dir, f"trainjson_{sketch_name}_{world}_frac{data_frac:.2f}_seed{subset_seed}.json")


def _cache_meta(sketch_name: str, world: str, data_frac: float, subset_seed: int,
                n_images: int, git_commit: str = "") -> dict:
    return {"logic_version": SUBSET_LOGIC_VERSION, "sketch_name": sketch_name,
            "world": world, "fraction": round(float(data_frac), 2),
            "subset_seed": int(subset_seed), "n_images": int(n_images),
            "git_commit": git_commit}


def cache_is_valid(cache_path: str, sketch_name: str, world: str,
                   data_frac: float, subset_seed: int) -> bool:
    """Cheap validity check via the tiny sidecar (never loads the big json)."""
    meta_path = cache_path + ".meta"
    if not (os.path.exists(cache_path) and os.path.exists(meta_path)):
        return False
    try:
        with open(meta_path) as f:
            m = json.load(f)
    except Exception:
        return False
    return (m.get("logic_version") == SUBSET_LOGIC_VERSION
            and m.get("sketch_name") == sketch_name
            and m.get("world") == world
            and round(m.get("fraction", -1), 2) == round(float(data_frac), 2)
            and m.get("subset_seed") == int(subset_seed))


def write_train_cache(cache_path: str, json_file: dict, sketch_name: str, world: str,
                      data_frac: float, subset_seed: int, git_commit: str = "") -> None:
    """Atomically write the filtered train-json + its sidecar stamp."""
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    meta = _cache_meta(sketch_name, world, data_frac, subset_seed,
                       len(json_file.get('images', [])), git_commit)
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(json_file, f)
    os.replace(tmp, cache_path)
    with open(cache_path + ".meta", "w") as f:
        json.dump(meta, f)
