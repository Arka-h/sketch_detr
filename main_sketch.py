# Sketch-DETR training/eval driver. Based on DETR's main.py; routes to models.sketch_detr,
# feeds sketches through engine_sketch, and scores eval with the §4 binary seed-14 GT.
# Vanilla main.py is left untouched (keeps the verified DETR-reproduction path).

import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from datasets.coco_sketch import build as build_sketch_dataset
from engine_sketch import train_one_epoch, evaluate, gt_calibration
from models.sketch_detr import build as build_sketch_model
from main import get_args_parser as detr_args


def get_args_parser():
    parser = argparse.ArgumentParser('Sketch-DETR', parents=[detr_args()], add_help=False)
    parser.add_argument('--sketch_cond', default='encoder_concat',
                        choices=['encoder_concat', 'object_query'])
    parser.add_argument('--sketch_ckpt', default='', help='frozen ζ backbone weights')
    parser.add_argument('--detr_init', default='checkpoints/detr-r50-e632da11.pth',
                        help='COCO-pretrained DETR to init from')
    parser.add_argument('--train_scheme_world', default='closed', choices=['closed', 'open'])
    parser.add_argument('--sketch_dataset', default='qd', choices=['qd', 'sketchy'])
    parser.add_argument('--num_sketches', default=1, type=int)
    parser.add_argument('--data_frac', default=1.0, type=float)
    parser.add_argument('--deterministic', action='store_true',
                        help='enable bit-deterministic eval (handover §4)')
    parser.add_argument('--gt_calib', action='store_true',
                        help='run GT-calibration probe on val seed-14 binary GT and exit')
    return parser


def set_determinism(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    if args.deterministic:
        set_determinism(args.seed)            # BEFORE model build (handover §4)
    else:
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    args.dataset_file = 'coco_sketch'
    dataset_val = build_sketch_dataset('val', args)
    base_ds = dataset_val.build_seed14_binary_gt()

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers)

    if args.gt_calib:
        stats = gt_calibration(data_loader_val, base_ds, device)
        print(f"[GT-CALIB] mAP={stats[0]:.4f} AP50={stats[1]:.4f} (expect ~1.0)")
        return

    model, criterion, postprocessors = build_sketch_model(args)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable params:', n_train)

    param_dicts = [{"params": [p for p in model_without_ddp.parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(ck['model'])
        if not args.eval and 'optimizer' in ck and 'epoch' in ck:
            optimizer.load_state_dict(ck['optimizer'])
            lr_scheduler.load_state_dict(ck['lr_scheduler'])
            args.start_epoch = ck['epoch'] + 1

    if args.eval:
        stats, _ = evaluate(model, postprocessors, data_loader_val, base_ds, device)
        print("[EVAL] coco_eval_bbox(12) =", [round(x, 4) for x in stats['coco_eval_bbox']])
        return

    dataset_train = build_sketch_dataset('train', args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    print("Start training")
    start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer,
                                      device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        stats, _ = evaluate(model, postprocessors, data_loader_val, base_ds, device)
        bb = stats['coco_eval_bbox']
        print(f"[epoch {epoch}] mAP={bb[0]:.4f} AP50={bb[1]:.4f} AP75={bb[2]:.4f} "
              f"AP_s={bb[3]:.4f} AP_m={bb[4]:.4f} AP_l={bb[5]:.4f}")
        if output_dir and utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'coco_eval_bbox': bb, 'epoch': epoch}
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            utils.save_on_master({'model': model_without_ddp.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'lr_scheduler': lr_scheduler.state_dict(),
                                  'epoch': epoch, 'args': args},
                                 output_dir / 'checkpoint.pth')
    print('Training time', str(datetime.timedelta(seconds=int(time.time() - start))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sketch-DETR', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
