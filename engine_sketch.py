# Train/eval loops for Sketch-DETR. Unpacks the (samples, targets, sketches) batch,
# feeds sketches into model(samples, sketches), and scores eval with the §4 binary
# seed-14 GT (one category per image, category_id=0, pycocotools 12-stat).

import math
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator


def _stack_sketches(sketches, device):
    return torch.stack([s for s in sketches], dim=0).to(device)


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm=0,
                    scaler=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 200

    for samples, targets, sketches in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]
        sketches = _stack_sketches(sketches, device)

        # AMP forward (autocast); frozen backbone/ζ/encoder + decoder/heads run fp16 on matmuls
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(samples, sketches)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training\n{loss_dict_reduced}")
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled,
                             class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, postprocessors, data_loader, base_ds, device, output_dir=None):
    """base_ds = §4 binary seed-14 GT (category_id=0). Returns (stats, coco_evaluator)."""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    coco_evaluator = CocoEvaluator(base_ds, ('bbox',))

    for samples, targets, sketches in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
        sketches = _stack_sketches(sketches, device)

        outputs = model(samples, sketches)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {t['image_id'].item(): out for t, out in zip(targets, results)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    stats = {}
    stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    return stats, coco_evaluator


@torch.no_grad()
def gt_calibration(data_loader, base_ds, device):
    """Sanity probe: feed GT boxes as perfect predictions (score 1.0) → expect mAP≈1.0.
    Catches category-id / coordinate-space mismatches before trusting a model's AP."""
    from util.box_ops import box_cxcywh_to_xyxy
    coco_evaluator = CocoEvaluator(base_ds, ('bbox',))
    seed14 = base_ds  # binary GT keyed by image_id, category 0
    img_to_anns = seed14.imgToAnns
    for img_id in seed14.getImgIds():
        anns = img_to_anns.get(img_id, [])
        if not anns:
            continue
        boxes = torch.tensor([a['bbox'] for a in anns], dtype=torch.float32)  # xywh abs
        xyxy = boxes.clone(); xyxy[:, 2:] += xyxy[:, :2]
        res = {img_id: {'scores': torch.ones(len(anns)),
                        'labels': torch.zeros(len(anns), dtype=torch.long),
                        'boxes': xyxy}}
        coco_evaluator.update(res)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator.coco_eval['bbox'].stats.tolist()
