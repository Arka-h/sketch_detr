# Sketch-DETR (Riba et al. 2021, "Localizing ∞-shaped fishes") on the vanilla DETR base.
#
# Dual backbone: image ψ (ResNet-50, from COCO DETR) + sketch ζ (ResNet-50 QD classifier,
# frozen). Two conditioning variants:
#   - encoder_concat (PRIMARY): tile f_s to HxW, concat with the d×H×W image feature,
#     1×1 conv back to d, then the transformer encoder.
#   - object_query: concat each object query with f_s, linear back to d.
# Binary class-agnostic Hungarian set loss (num_classes=1 → foreground=0, no-object=1).
# Init from COCO-pretrained DETR; freeze image backbone, sketch backbone, encoder.

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50

from util.misc import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer import build_transformer
from .matcher import build_matcher
from .detr import SetCriterion, PostProcess, MLP


class SketchEncoder(nn.Module):
    """Frozen ResNet-50 sketch classifier ζ → f_s ∈ R^2048 (global-pooled)."""

    def __init__(self, ckpt_path=None):
        super().__init__()
        self.body = resnet50(weights=None)
        self.body.fc = nn.Identity()      # forward → (B,2048) after avgpool+flatten
        self.feat_dim = 2048
        if ckpt_path:
            ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # format-agnostic: accept backbone_state | model_state | model | state_dict | raw,
            # strip any 'module.' (DDP) prefix and the classifier head (fc.*).
            val_acc = ck.get('val_acc') if isinstance(ck, dict) else None
            if isinstance(ck, dict):
                for key in ('backbone_state', 'model_state', 'model', 'state_dict'):
                    if key in ck and isinstance(ck[key], dict):
                        ck = ck[key]; break
            bs = {(k[len('module.'):] if k.startswith('module.') else k): v
                  for k, v in ck.items()}
            bs = {k: v for k, v in bs.items() if not k.startswith('fc.')}
            ref = {k for k in self.body.state_dict() if not k.startswith('fc.')}
            matched = {k for k in bs if k in ref}
            missing, unexpected = self.body.load_state_dict(bs, strict=False)
            missing = [k for k in missing if not k.startswith('fc.')]
            print(f"[ζ] loaded {ckpt_path}: matched={len(matched)}/{len(ref)} "
                  f"missing={len(missing)} unexpected={len(unexpected)} val_acc={val_acc}")
            assert len(missing) == 0 and len(matched) == len(ref), (
                f"ζ load incomplete: matched {len(matched)}/{len(ref)}, missing {missing[:5]} "
                f"— wrong/incompatible checkpoint?")

    @torch.no_grad()
    def forward(self, sketches):  # (B,3,H,W) → (B,2048)
        return self.body(sketches)


class SketchDETR(nn.Module):
    def __init__(self, backbone, transformer, sketch_encoder, num_classes, num_queries,
                 sketch_cond='encoder_concat', aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.sketch_cond = sketch_cond

        # sketch conditioning
        self.sketch_encoder = sketch_encoder
        self.sketch_proj = nn.Linear(sketch_encoder.feat_dim, hidden_dim)
        if sketch_cond == 'encoder_concat':
            self.fuse_conv = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
        elif sketch_cond == 'object_query':
            self.query_fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            raise ValueError(f'unknown sketch_cond {sketch_cond}')

    def forward(self, samples: NestedTensor, sketches):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        src = self.input_proj(src)                       # (B,d,H,W)
        pos_embed = pos[-1]
        bs, d, h, w = src.shape

        f_s = self.sketch_proj(self.sketch_encoder(sketches))   # (B,d)

        if self.sketch_cond == 'encoder_concat':
            tile = f_s[:, :, None, None].expand(-1, -1, h, w)   # (B,d,H,W)
            src = self.fuse_conv(torch.cat([src, tile], dim=1)) # (B,d,H,W)
            query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # (Q,B,d)
        else:  # object_query
            q = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)        # (B,Q,d)
            fs_q = f_s.unsqueeze(1).expand(-1, self.num_queries, -1)           # (B,Q,d)
            q = self.query_fuse(torch.cat([q, fs_q], dim=-1))                  # (B,Q,d)
            query_pos = q.permute(1, 0, 2)                                     # (Q,B,d)

        # run transformer encoder/decoder directly (keeps vanilla internals intact)
        src_f = src.flatten(2).permute(2, 0, 1)          # (HW,B,d)
        pos_f = pos_embed.flatten(2).permute(2, 0, 1)
        mask_f = mask.flatten(1)
        memory = self.transformer.encoder(src_f, src_key_padding_mask=mask_f, pos=pos_f)
        tgt = torch.zeros_like(query_pos)
        hs = self.transformer.decoder(tgt, memory, memory_key_padding_mask=mask_f,
                                      pos=pos_f, query_pos=query_pos)
        hs = hs.transpose(1, 2)                           # (layers,B,Q,d)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                  for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return out


def _freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)


def build(args):
    device = torch.device(args.device)
    num_classes = 1  # binary: foreground=0, no-object=1

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    sketch_encoder = SketchEncoder(getattr(args, 'sketch_ckpt', None))

    model = SketchDETR(
        backbone, transformer, sketch_encoder,
        num_classes=num_classes, num_queries=args.num_queries,
        sketch_cond=getattr(args, 'sketch_cond', 'encoder_concat'),
        aux_loss=args.aux_loss,
    )

    # init from COCO-pretrained DETR (skip the 92-way class head)
    init_ckpt = getattr(args, 'detr_init', None)
    if init_ckpt:
        ck = torch.load(init_ckpt, map_location='cpu', weights_only=False)
        sd = ck['model'] if 'model' in ck else ck
        sd = {k: v for k, v in sd.items() if not k.startswith('class_embed.')}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        miss_new = [k for k in missing if not (k.startswith('sketch_encoder.') or
                    k.startswith('sketch_proj.') or k.startswith('fuse_conv.') or
                    k.startswith('query_fuse.') or k.startswith('class_embed.'))]
        print(f"[init] DETR COCO weights loaded. unexpected={len(unexpected)} "
              f"non-new-missing={miss_new[:6]}")

    # freeze image backbone ψ, sketch backbone ζ, transformer encoder
    _freeze(model.backbone)
    _freeze(model.sketch_encoder)
    _freeze(model.transformer.encoder)

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    if args.aux_loss:
        aux = {}
        for i in range(args.dec_layers - 1):
            aux.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux)
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    return model, criterion, postprocessors
