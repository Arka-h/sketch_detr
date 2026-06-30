#!/usr/bin/env python
"""Edge-case test suite for the seeded/nested/leak-free subset selection.

This is the REFERENCE proof of the fix (datasets/subset_select.py). Every affected
repo gets the same subset_select.py and must pass an equivalent check. Run:

    PYTHONPATH=. python tests/test_subset_select.py
    PYTHONPATH=. python tests/test_subset_select.py --coco /mnt/1tb/data/coco/annotations/instances_train2017.json

Covers: determinism, nesting (0.25⊂0.50⊂1.00), replace=False exact fraction,
ZERO held-out leakage at EVERY fraction, closed-world (no unseen), duplicate
image-ids, tiny-fraction behaviour, seed sensitivity, cache-key versioning +
auto-invalidation, and persist write-once/verify-mismatch.
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

from datasets.subset_select import (
    SUBSET_LOGIC_VERSION, open_world_split, collect_image_ids,
    select_nested_subset_ids, finalize_train_ids,
    persist_subset_bundle, train_cache_path, cache_is_valid, write_train_cache,
)

PASS, FAIL = "✅", "❌"
_fails = []


def check(name, cond):
    print(f"  {PASS if cond else FAIL} {name}")
    if not cond:
        _fails.append(name)


# --- synthetic fixture: cats with duplicate img-ids + a shared seen/unseen image ---
def fixture():
    # seen cats 1,2,3 ; unseen (held-out) cats 9,10
    seen = {
        1: [10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],   # dup of 10; img 12 also unseen
        2: list(range(100, 140)),                              # 40 unique
        3: [5, 5, 6, 7],                                       # 3 unique
    }
    unseen = {9: [12, 200, 201, 202], 10: [300, 301]}          # img 12 shared with seen cat 1
    return seen, unseen


def all_unseen(unseen):
    return set().union(*unseen.values()) if unseen else set()


def test_core():
    print("[core selection]")
    seen, unseen = fixture()
    U = all_unseen(unseen)

    f25, c25 = finalize_train_ids(seen, unseen, 0.25, 14)
    f50, _ = finalize_train_ids(seen, unseen, 0.50, 14)
    f100, _ = finalize_train_ids(seen, unseen, 1.00, 14)

    # determinism
    g25, _ = finalize_train_ids(seen, unseen, 0.25, 14)
    check("deterministic (same seed -> identical)", f25 == g25)

    # nesting
    check("nested 0.25 ⊂ 0.50", f25 <= f50)
    check("nested 0.50 ⊂ 1.00", f50 <= f100)

    # ZERO leakage at every fraction (the critical property)
    check("0 held-out leak @0.25", len(f25 & U) == 0)
    check("0 held-out leak @0.50", len(f50 & U) == 0)
    check("0 held-out leak @1.00", len(f100 & U) == 0)
    check("img 12 (shared seen+unseen) excluded at all fracs",
          12 not in f25 and 12 not in f50 and 12 not in f100)

    # replace=False exact fraction (cat 2: 40 unique -> int(0.25*40)=10)
    sel = select_nested_subset_ids(seen, 0.25, 14)
    check("replace=False exact: cat2 @0.25 == 10 unique", len(sel[2]) == 10)
    check("replace=False exact: cat1 @0.25 == int(0.25*11)=2", len(sel[1]) == 2)  # 11 unique (dedup 10)
    check("no duplicates within a category selection", all(len(v) == len(set(v)) for v in sel.values()))

    # 1.00 keeps ALL seen (minus shared-unseen img 12)
    seen_all = set().union(*seen.values())
    check("1.00 == all seen − unseen", f100 == (seen_all - U))

    # seed sensitivity
    h25, _ = finalize_train_ids(seen, unseen, 0.25, 99)
    check("seed-sensitive (diff seed -> diff set, same size)", h25 != f25 and len(h25) == len(f25))


def test_closed_world():
    print("[closed-world / empty unseen]")
    seen, _ = fixture()
    f, c = finalize_train_ids(seen, {}, 0.50, 14)   # no unseen dict
    seen_all = set().union(*seen.values())
    exp = set().union(*select_nested_subset_ids(seen, 0.50, 14).values())
    check("empty unseen -> no crash, final == subset(seen)", f == exp)
    check("closed per-class counts present", len(c) == 3)


def test_tiny_fraction():
    print("[tiny fraction edge]")
    seen = {1: list(range(50)), 2: list(range(50, 53))}  # cat2 only 3 imgs
    f_small = select_nested_subset_ids(seen, 0.01, 14)
    # int(0.01*50)=0 and int(0.01*3)=0 -> categories can zero out at extreme fracs (documented)
    check("extreme tiny frac zeroes out small cats (documented, no crash)",
          len(f_small[1]) == 0 and len(f_small[2]) == 0)
    # still nests into larger fracs
    f5 = select_nested_subset_ids(seen, 0.5, 14)
    check("tiny ⊂ larger still holds", f_small[1] <= f5[1])


def test_cache_versioning():
    print("[cache key versioning + invalidation]")
    with tempfile.TemporaryDirectory() as d:
        p = train_cache_path(d, "qd", "open", 0.25, 14)
        jf = {"images": [{"id": 1}], "annotations": [], "categories": []}
        write_train_cache(p, jf, "qd", "open", 0.25, 14, git_commit="abc")
        check("written cache is valid for same key", cache_is_valid(p, "qd", "open", 0.25, 14))
        check("invalid for different seed", not cache_is_valid(p, "qd", "open", 0.25, 99))
        check("invalid for different fraction", not cache_is_valid(p, "qd", "open", 0.50, 14))
        check("invalid for different world", not cache_is_valid(p, "qd", "closed", 0.25, 14))
        # tamper the sidecar logic_version -> must invalidate
        meta = json.load(open(p + ".meta"))
        meta["logic_version"] = "OLD-vN"
        json.dump(meta, open(p + ".meta", "w"))
        check("stale logic_version auto-invalidates", not cache_is_valid(p, "qd", "open", 0.25, 14))
        check("SUBSET_LOGIC_VERSION is set", bool(SUBSET_LOGIC_VERSION))


def test_persist_guard():
    print("[persist write-once + verify-mismatch]")
    with tempfile.TemporaryDirectory() as d:
        id2c = {1: "cat1"}
        p1 = persist_subset_bundle(d, "qd", "open", 0.25, 14, {1, 2, 3}, {1: 3}, id2c)
        # same ids -> ok (idempotent)
        persist_subset_bundle(d, "qd", "open", 0.25, 14, {1, 2, 3}, {1: 3}, id2c)
        check("idempotent re-persist of identical ids", os.path.exists(p1))
        # different ids, same key -> must RAISE (the reproducibility tripwire)
        raised = False
        try:
            persist_subset_bundle(d, "qd", "open", 0.25, 14, {1, 2, 999}, {1: 3}, id2c)
        except RuntimeError:
            raised = True
        check("mismatched ids for same key RAISES (no silent overwrite)", raised)


def test_real_coco(coco_json):
    print(f"[real COCO: {os.path.basename(coco_json)}]")
    ALL = ['bicycle','car','airplane','bus','train','truck','traffic light','fire hydrant','stop sign','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','suitcase','baseball bat','skateboard','wine glass','cup','fork','knife','spoon','banana','apple','sandwich','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','bed','toilet','laptop','mouse','keyboard','cell phone','microwave','oven','toaster','sink','book','clock','vase','scissors','toothbrush']
    _, unseen_cats = open_world_split(ALL)
    coco = json.load(open(coco_json))
    id2c = {c['id']: c['name'] for c in coco['categories']}
    _, seen, unseen = collect_image_ids(coco, id2c, ALL, unseen_cats, 'train', 'open')
    U = set().union(*unseen.values())
    sets = {fr: finalize_train_ids(seen, unseen, fr, 14)[0] for fr in (0.25, 0.50, 1.00)}
    check("real: nested 0.25⊂0.50⊂1.00", sets[0.25] <= sets[0.50] <= sets[1.00])
    for fr in (0.25, 0.50, 1.00):
        check(f"real: 0 Set-B leak @{fr}", len(sets[fr] & U) == 0)
    # determinism across a fresh collect+finalize
    _, seen2, unseen2 = collect_image_ids(coco, id2c, ALL, unseen_cats, 'train', 'open')
    again = finalize_train_ids(seen2, unseen2, 0.25, 14)[0]
    check("real: deterministic across rebuild", again == sets[0.25])
    print(f"    sizes: 25%={len(sets[0.25])} 50%={len(sets[0.50])} 100%={len(sets[1.00])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco', default='')
    args = ap.parse_args()
    test_core()
    test_closed_world()
    test_tiny_fraction()
    test_cache_versioning()
    test_persist_guard()
    if args.coco and os.path.exists(args.coco):
        test_real_coco(args.coco)
    print()
    if _fails:
        print(f"{FAIL} {len(_fails)} FAILED: {_fails}")
        sys.exit(1)
    print(f"{PASS} ALL TESTS PASSED")


if __name__ == '__main__':
    main()
