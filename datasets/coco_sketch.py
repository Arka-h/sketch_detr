# Sketch-conditioned COCO dataset for the Sketch-DETR baseline (Riba et al. 2021).
#
# Ported from the thesis code (clip_ddetr_ow_repr / clean_run) for SetB / holdout /
# dataloader / seed-14 protocol ONLY. Everything CASF-specific (CLIP text/image
# embeddings, CLIP normalisation) is intentionally stripped: Sketch-DETR conditions
# on a CNN sketch-classifier embedding f_s, not on CLIP.
#
# A "query" = (natural image, one sketch of one category). Target = all instances of
# that category in the image (binary, class-agnostic: foreground label 0, no-object 1).
#
# Returns per item: (img, target, sketch)  where sketch is a single CHW tensor.
# DETR's collate_fn zips these; the engine stacks the sketch tuple into (B,C,H,W).

import json
import os
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from tqdm import tqdm

import datasets.transforms as T
from datasets.coco import ConvertCocoPolysToMask
from util.misc import get_rank, is_dist_avail_and_initialized
import torch.distributed as dist


# ── sketch rendering ───────────────────────────────────────────────────────────

def rasterize_stroke3(strokes, size: int = 224, line_width: int = 2, padding: int = 10):
    """Render a QuickDraw stroke-3 array (dx, dy, pen_up) as a white-on-black PIL image.
    Same convention as the thesis QD loader (sketchrnn arrays)."""
    pts = np.asarray(strokes, dtype=np.float32)
    # absolute coordinates from deltas
    abs_xy = np.cumsum(pts[:, :2], axis=0)
    # split into strokes on pen-up (column index 2 == 1 ends a stroke)
    lifts = np.nonzero(pts[:, 2])[0]
    img = Image.new('RGB', (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    if len(abs_xy) == 0:
        return img
    mn = abs_xy.min(axis=0)
    mx = abs_xy.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    scale = (size - 2 * padding) / span.max()
    norm = (abs_xy - mn) * scale + padding
    start = 0
    for end in list(lifts) + [len(norm) - 1]:
        seg = norm[start:end + 1]
        if len(seg) >= 2:
            draw.line([tuple(p) for p in seg], fill=(255, 255, 255), width=line_width)
        start = end + 1
    return img


# ── base sketch dataset ──────────────────────────────────────────────────────────

class CocoDetectionSketch(torchvision.datasets.CocoDetection):
    """COCO + sketch-query base. Builds the open/closed-world category split
    (every-4th-category i%4==0 holdout = SetB on the QD order), the COCO subset
    temp-json, and the __getitem__ sketch pipeline. Subclasses provide:
        ALL_CATEGORIES   (class attr) — ordered category list (load-bearing)
        sketch_name      (property)   — 'qd' | 'sk'
        _setup_sketches()             — populate self.class2quick {cat: [refs]}
        _process_sketch(ref) -> Tensor — render one ref to a CHW tensor
    """

    ALL_CATEGORIES: List[str] = []

    @property
    def sketch_name(self) -> str:
        raise NotImplementedError

    def _setup_sketches(self) -> None:
        raise NotImplementedError

    def _process_sketch(self, ref: str) -> torch.Tensor:
        raise NotImplementedError

    def __init__(self, image_set, img_folder, ann_file, transforms, return_masks,
                 data_frac=1.0, train_scheme_world="open", sketch_root=None,
                 num_sketches=1, seed14=14):
        self.image_set = image_set
        self.data_frac = data_frac
        self.train_scheme_world = train_scheme_world
        self.num_sketches = num_sketches
        self.seed14 = seed14
        self.sketch_root = sketch_root if sketch_root is not None \
            else os.environ.get('SKETCH_HOME', '/mnt/1tb/data')

        with open(ann_file) as f:
            json_file = json.load(f)

        self.id2class, self.class2id = {}, {}
        for cat in json_file['categories']:
            self.id2class[cat['id']] = cat['name']
            self.class2id[cat['name']] = cat['id']
        self.all_categories = list(self.ALL_CATEGORIES)

        # open-world holdout: leave out every 4th category (SetB), train on the rest
        if self.train_scheme_world == "open":
            unseen = [self.all_categories[i] for i in range(len(self.all_categories)) if i % 4 == 0]
            seen = [self.all_categories[i] for i in range(len(self.all_categories)) if i % 4 != 0]
        elif self.train_scheme_world == "closed":
            unseen, seen = [], list(self.all_categories)
        else:
            raise ValueError(f"Unknown training scheme: {self.train_scheme_world}")
        self.unseen_cats, self.seen_cats = list(unseen), list(seen)

        # which categories' instances are visible in this split:
        #   train (any scheme) and closed-val: all seen categories
        #   open-val: only the held-out (unseen) categories
        if self.image_set == 'val' and self.train_scheme_world == 'open':
            visible = set(unseen)
        else:
            visible = set(seen)
        self.visible_cats = visible

        annotate, seen_image_ids = [], {}
        for anno in json_file['annotations']:
            cname = self.id2class[anno['category_id']]
            if cname in visible:
                annotate.append(anno)
                seen_image_ids.setdefault(anno['category_id'], []).append(anno['image_id'])

        if self.image_set == 'train' and self.data_frac < 1.0:
            rng = np.random.RandomState(0)
            sub = {k: set(rng.choice(v, int(self.data_frac * len(v)), replace=False).tolist())
                   for k, v in seen_image_ids.items()}
            seen_image_ids = sub
            keep_imgs = set().union(*seen_image_ids.values()) if seen_image_ids else set()
            annotate = [a for a in annotate if a['image_id'] in keep_imgs]

        keep_imgs = set().union(*seen_image_ids.values()) if seen_image_ids else set()
        images = [im for im in json_file['images'] if im['id'] in keep_imgs]
        json_file['annotations'] = annotate
        json_file['images'] = images
        print(f"[holdout] scheme={train_scheme_world} set={image_set} "
              f"visible_cats={len(visible)} imgs={len(images)} anns={len(annotate)} "
              f"unseen={len(unseen)} seen={len(seen)}")

        os.makedirs('annotations', exist_ok=True)
        temp_ann_file = os.path.join(
            f'annotations/temp_json_{self.sketch_name}_{image_set}_{train_scheme_world}.json')
        if get_rank() == 0:
            with open(temp_ann_file, 'w') as f:
                json.dump(json_file, f)
        if is_dist_avail_and_initialized():
            dist.barrier()

        super().__init__(img_folder, temp_ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

        # sketch preprocessing: ImageNet stats (ζ is an ImageNet-init ResNet-50)
        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transforms_sketch = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), normalize])

        self.class2quick = defaultdict(list)
        self._setup_sketches()

        # deterministic seed-14 category map for val: one category per image, robust
        # (sorted candidate list + fresh Random(seed14) per image — NOT global seed).
        self._seed14_cat_map = {}
        if self.image_set == 'val':
            self._build_seed14_map()

    # ── seed-14 protocol ──────────────────────────────────────────────────────
    def _build_seed14_map(self):
        """For each val image, pick exactly one present visible category as the query,
        deterministically. Only categories that have sketches are eligible."""
        for img_id in self.ids:
            anns = self.coco.imgToAnns.get(img_id, [])
            cats = sorted({a['category_id'] for a in anns
                           if self.id2class[a['category_id']] in self.visible_cats
                           and self.class2quick.get(self.id2class[a['category_id']])})
            if not cats:
                continue
            self._seed14_cat_map[img_id] = random.Random(self.seed14).choice(cats)

    def build_seed14_binary_gt(self):
        """Return a pycocotools COCO holding binary GT (category_id=0) over the
        seed-14-selected category per val image — the §4 scoring ground truth."""
        from pycocotools.coco import COCO
        gt = {'images': [], 'annotations': [], 'categories': [{'id': 0, 'name': 'object'}]}
        ann_id = 1
        for img in self.coco.dataset['images']:
            img_id = img['id']
            sel = self._seed14_cat_map.get(img_id)
            if sel is None:
                continue
            gt['images'].append(img)
            for a in self.coco.imgToAnns.get(img_id, []):
                if a['category_id'] != sel:
                    continue
                gt['annotations'].append({
                    'id': ann_id, 'image_id': img_id, 'category_id': 0,
                    'bbox': a['bbox'], 'area': a['area'],
                    'iscrowd': a.get('iscrowd', 0)})
                ann_id += 1
        coco = COCO()
        coco.dataset = gt
        coco.createIndex()
        return coco

    # ── item access ───────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img, anns = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': anns}
        img, target = self.prepare(img, target)

        present = sorted(set(target['labels'].tolist()))
        # restrict to categories that have sketches
        present = [c for c in present
                   if self.class2quick.get(self.id2class[c]) and self.id2class[c] in self.visible_cats]

        if self.image_set == 'val':
            selected_cat = self._seed14_cat_map.get(image_id, present[0] if present else target['labels'][0].item())
        else:
            rng = random.Random(int(1000 * time.time()) ^ (idx * 2654435761))
            selected_cat = rng.choice(present) if present else target['labels'][0].item()

        keep = target['labels'] == selected_cat
        new_target = {}
        for key, value in target.items():
            if key in ('boxes', 'labels', 'area', 'iscrowd', 'masks'):
                new_target[key] = value[keep]
            else:
                new_target[key] = value
        # binary class-agnostic: foreground label 0 (num_classes=1 → no-object index 1)
        new_target['labels'] = torch.zeros_like(new_target['labels'])

        cat_name = self.id2class[selected_cat]
        # sketch query: single sketch (k=1) for val (deterministic); train samples one
        refs = (random.Random(self.seed14).sample(self.class2quick[cat_name], 1)
                if self.image_set == 'val'
                else random.choices(self.class2quick[cat_name], k=self.num_sketches))
        sketch = self._process_sketch(refs[0])  # k=1 single query (handover §4)

        old_boxes = new_target['boxes'].clone()
        if self._transforms is not None:
            img, new_target = self._transforms(img, new_target)
        if self.image_set == 'val':
            new_target['boxes'] = old_boxes  # keep abs boxes for eval; postproc rescales
        new_target['query_cat_id'] = torch.tensor([selected_cat])
        return img, new_target, sketch


class CocoDetectionQD(CocoDetectionSketch):
    """QuickDraw sketch queries: stroke-3 arrays rasterised white-on-black."""

    ALL_CATEGORIES = ['bicycle', 'car', 'airplane', 'bus', 'train', 'truck', 'traffic light',
                      'fire hydrant', 'stop sign', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                      'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'suitcase',
                      'baseball bat', 'skateboard', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                      'banana', 'apple', 'sandwich', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                      'cake', 'chair', 'couch', 'bed', 'toilet', 'laptop', 'mouse', 'keyboard',
                      'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'book', 'clock', 'vase',
                      'scissors', 'toothbrush']

    @property
    def sketch_name(self):
        return 'qd'

    def _setup_sketches(self):
        qd_root = os.path.join(self.sketch_root, 'quickdraw', 'sketchrnn')
        split = 'train' if self.image_set == 'train' else 'valid'
        print(f"Loading Quick,Draw! ({split}) from {qd_root} ...")
        self._qd_mmap = {}
        for cat in self.all_categories:
            ptr_p = os.path.join(qd_root, f'{cat}.{split}.ptr.npy')
            stk_p = os.path.join(qd_root, f'{cat}.{split}.strokes.npy')
            if not (os.path.exists(ptr_p) and os.path.exists(stk_p)):
                print(f"[qd] WARN missing sketchrnn arrays for cat={cat!r} split={split}; skipping")
                continue
            ptr = np.load(ptr_p)
            strokes = np.load(stk_p, mmap_mode='r')
            self._qd_mmap[cat] = (ptr, strokes)
            self.class2quick[cat] = [f'{cat}:{i}' for i in range(len(ptr) - 1)]

    def _process_sketch(self, ref):
        cat_name, idx_str = ref.rsplit(':', 1)
        ptr, strokes = self._qd_mmap[cat_name]
        i_ = int(idx_str)
        sketch = rasterize_stroke3(strokes[ptr[i_]:ptr[i_ + 1]])
        return self.transforms_sketch(sketch)


class CocoDetectionSketchy(CocoDetectionSketch):
    """Sketchy sketch queries: PNGs inverted to white-on-black, resized to 224."""

    ALL_CATEGORIES = ['elephant', 'bear', 'cat', 'zebra', 'horse', 'giraffe', 'airplane',
                      'dog', 'scissors', 'pizza', 'cow', 'umbrella', 'sheep', 'bicycle',
                      'hot dog', 'banana', 'couch', 'bench', 'chair', 'apple', 'cup',
                      'car', 'knife', 'clock', 'spoon', 'mouse', 'motorcycle']

    @property
    def sketch_name(self):
        return 'sk'

    def _setup_sketches(self):
        sk_root = os.path.join(self.sketch_root, 'sketchy')
        split = 'train' if self.image_set == 'train' else 'test'
        print(f"Loading Sketchy ({split}) from {sk_root} ...")
        with open(os.path.join(sk_root, 'sketchy_dataset.pkl'), 'rb') as f:
            data = pickle.load(f)
        img_dir = os.path.join(sk_root, 'images')
        for cat in self.all_categories:
            stems = data[split].get(cat, [])
            if not stems:
                print(f"[sk] WARN no Sketchy sketches for cat={cat!r} split={split}; skipping")
                continue
            self.class2quick[cat] = [os.path.join(img_dir, f'{stem}.png') for stem in stems]

    def _process_sketch(self, ref):
        im = Image.open(ref).convert('RGB')
        im = Image.fromarray(255 - np.array(im))  # black-on-white -> white-on-black
        im = im.resize((224, 224))
        return self.transforms_sketch(im)


def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
    if image_set == 'val':
        return T.Compose([T.RandomResize([800], max_size=1333), normalize])
    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / 'instances_train2017.json'),
        "val": (root / "val2017", root / "annotations" / 'instances_val2017.json'),
    }
    img_folder, ann_file = PATHS[image_set]
    kw = dict(transforms=make_coco_transforms(image_set), return_masks=args.masks,
              data_frac=getattr(args, 'data_frac', 1.0),
              train_scheme_world=getattr(args, 'train_scheme_world', 'closed'),
              num_sketches=getattr(args, 'num_sketches', 1))
    cls = CocoDetectionSketchy if getattr(args, 'sketch_dataset', 'qd') == 'sketchy' else CocoDetectionQD
    return cls(image_set, img_folder, ann_file, **kw)
