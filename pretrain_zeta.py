# Pretrain the sketch backbone ζ: a ResNet-50 sketch classifier over the COCO-intersecting
# QuickDraw classes. Two configs:
#   --classes cw56  → all 56 COCO∩QD classes      (Job A closed-world)
#   --classes ow42  → 42 seen (exclude Set B)      (Job B open-world; Set B provably excluded)
# Saves {backbone_state (resnet50 minus fc), classes, val_acc} → loadable by SketchEncoder.

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from datasets.coco_sketch import rasterize_stroke3, CocoDetectionQD

# Set B = i%4==0 on the QD order (the open-world holdout)
QD_ALL = CocoDetectionQD.ALL_CATEGORIES
SET_B = sorted({QD_ALL[i] for i in range(len(QD_ALL)) if i % 4 == 0})


def get_classes(which):
    if which == 'cw56':
        return list(QD_ALL)
    if which == 'ow42':
        return [c for c in QD_ALL if c not in SET_B]
    raise ValueError(which)


class QDClassify(Dataset):
    def __init__(self, classes, split, sketch_root, per_class_cap, train, seed=0):
        self.classes = classes
        self.cls2idx = {c: i for i, c in enumerate(classes)}
        self.train = train
        qd_root = os.path.join(sketch_root, 'quickdraw', 'sketchrnn')
        self.mmaps = {}
        self.index = []  # (class_name, sketch_idx)
        rng = np.random.RandomState(seed)
        for c in classes:
            ptr = np.load(os.path.join(qd_root, f'{c}.{split}.ptr.npy'))
            stk = np.load(os.path.join(qd_root, f'{c}.{split}.strokes.npy'), mmap_mode='r')
            self.mmaps[c] = (ptr, stk)
            n = len(ptr) - 1
            ids = np.arange(n)
            if per_class_cap and n > per_class_cap:
                ids = rng.choice(n, per_class_cap, replace=False)
            self.index.extend((c, int(i)) for i in ids)
        norm = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        aug = [torchvision.transforms.RandomHorizontalFlip()] if train else []
        self.tf = torchvision.transforms.Compose(aug + [torchvision.transforms.ToTensor(), norm])
        print(f"[{split}] {len(self.index)} sketches over {len(classes)} classes "
              f"(cap={per_class_cap})")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        c, si = self.index[i]
        ptr, stk = self.mmaps[c]
        img = rasterize_stroke3(stk[ptr[si]:ptr[si + 1]])
        return self.tf(img), self.cls2idx[c]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', choices=['cw56', 'ow42'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sketch_root', default=os.environ.get('SKETCH_HOME', '/mnt/1tb/data'))
    ap.add_argument('--per_class_cap', type=int, default=12000)
    ap.add_argument('--val_cap', type=int, default=500)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--seed', type=int, default=14)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda'
    classes = get_classes(args.classes)
    if args.classes == 'ow42':
        assert not (set(classes) & set(SET_B)), "Set B leaked into ζ42!"
    print(f"ζ-{args.classes}: {len(classes)} classes. SetB excluded={args.classes=='ow42'}")

    tr = QDClassify(classes, 'train', args.sketch_root, args.per_class_cap, train=True, seed=args.seed)
    va = QDClassify(classes, 'valid', args.sketch_root, args.val_cap, train=False, seed=args.seed)
    trl = DataLoader(tr, args.batch_size, shuffle=True, num_workers=args.workers,
                     pin_memory=True, drop_last=True, persistent_workers=True)
    val = DataLoader(va, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    crit = nn.CrossEntropyLoss()

    best = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for x, y in trl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item(); nb += 1
        sched.step()
        acc = evaluate(model, val, device)
        print(f"epoch {ep+1}/{args.epochs} loss {run/max(nb,1):.3f} val_acc {acc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if acc >= best:
            best = acc
            bb = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith('fc.')}
            torch.save({'backbone_state': bb, 'arch': 'resnet50', 'feat_dim': 2048,
                        'classes': classes, 'val_acc': acc, 'config': args.classes,
                        'set_b_excluded': args.classes == 'ow42'}, args.out)
            print(f"  saved best -> {args.out} (val_acc {acc:.4f})", flush=True)
    print(f"DONE ζ-{args.classes}: best val_acc {best:.4f} -> {args.out}")


if __name__ == '__main__':
    main()
