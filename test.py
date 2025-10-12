#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Tuple
from tqdm import tqdm

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CochAV heatmap hit rate evaluation')
    parser.add_argument('--config', type=str, default='AudioCOCO/config6.json', help='Configuration JSON')
    parser.add_argument('--condition', type=str, default='normal', help='Silent condition')
    parser.add_argument('--label', type=str, default='no', help='Label')
    parser.add_argument('--image_root', type=str, default='/home/yanhao/coco/val2014/', help='Image root directory')
    parser.add_argument('--coch_root', type=str, default='/data/data0/coch/', help='coch .npy root directory')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader thread count')
    parser.add_argument('--gpu', type=str, default='3', help='GPU id, e.g. 0 or 0,1')
    parser.add_argument('--pretrained_path', type=str, required=True, help='cochAV pretrained weight path (.pth/.tar)')
    parser.add_argument('--neg', action='store_true', help='Enable Neg branch (must be consistent with training)')
    parser.add_argument('--tri_map', action='store_true', help='Enable Trimap (must be consistent with training)')
    parser.add_argument('--epsilon', type=float, default=0.65)
    parser.add_argument('--epsilon2', type=float, default=0.4)
    return parser.parse_args()


def setup_device(gpu: str) -> torch.device:
    if torch.cuda.is_available():
        ids = [int(x) for x in gpu.split(',') if x.strip() != '']
        torch.cuda.set_device(ids[0])
        return torch.device(f'cuda:{ids[0]}')
    return torch.device('cpu')


def main() -> None:
    args = parse_args()
    device = setup_device(args.gpu)

    # Lazy import to avoid unnecessary dependencies
    from AudioCOCO.dataset import create_npy_dataloader
    from models.CochAV import CochAV

    # Adapt parameter shape from training script
    class EvalArgs:
        def __init__(self, ns: argparse.Namespace):
            self.epsilon = ns.epsilon
            self.epsilon2 = ns.epsilon2
            self.tri_map = ns.tri_map
            self.Neg = ns.neg
            self.img_size = ns.img_size
            self.pretrained_path = ns.pretrained_path
            self.gpu_ids = [int(x) for x in ns.gpu.split(',') if x.strip() != ''] or [0]

    eval_args = EvalArgs(args)

    # DataLoader
    _, dataset = create_npy_dataloader(
        config_json_path=args.config,
        image_root=args.image_root,
        coch_root=args.coch_root,
        img_size=args.img_size,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        train=False,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = CochAV(eval_args, pretrained_path=eval_args.pretrained_path).to(device)
    model.eval()

    correct = 0
    total = 0
    # Statistics by size
    size_stats = {
        'size1': {'correct': 0, 'total': 0},
        'size2': {'correct': 0, 'total': 0},
        'size3': {'correct': 0, 'total': 0},
    }

    with torch.no_grad():
        for image_t, audio_coch_t, gt, _, _ in tqdm(loader, total=len(dataset), desc='Evaluating'):
            image_t = image_t.to(device, non_blocking=True)
            audio_coch_t = audio_coch_t.to(device, non_blocking=True)
            if args.config.endswith('config4.json'):
                if args.label == 'noise':
                    image_t = torch.randn_like(image_t)
                else:
                    image_t = torch.zeros_like(image_t)
            if args.condition == 'silent':
                if args.label == 'noise':
                    audio_coch_t = torch.randn_like(audio_coch_t)
                else:
                    audio_coch_t = torch.zeros_like(audio_coch_t)

            A, _, _, _, _ = model(image_t, audio_coch_t, eval_args, mode='val')
            heatmap = A[0, 0]  # [H, W]

            # Maximum value coordinates
            max_idx = torch.argmax(heatmap)
            h, w = heatmap.shape
            max_y = (max_idx // w).item()
            max_x = (max_idx % w).item()

            # GT bbox (already xyxy at 224 scale, long)
            bbox_xyxy = gt['bbox_xyxy_224'][0].to(device).long()
            xmin, ymin, xmax, ymax = [int(v.item()) for v in bbox_xyxy]

            # heatmap scale -> 224 scale coordinate alignment
            # ResNet18 outputs 14x14 feature map, need to map to 224x224 image
            target_size = args.img_size  # 224
            scale_x = target_size / float(w)  # 224/14 = 16
            scale_y = target_size / float(h)  # 224/14 = 16
            peak_x_224 = int(round((max_x + 0.5) * scale_x))
            peak_y_224 = int(round((max_y + 0.5) * scale_y))

            # Hit judgment: whether peak falls into gt_box
            hit = (xmin <= peak_x_224 <= xmax) and (ymin <= peak_y_224 <= ymax)
            correct += int(hit)
            total += 1

            # Get object_size grouping from config entry (compatible with different collate forms)
            try:
                meta = gt['meta']
                obj_size = None
                if isinstance(meta, dict):
                    if 'object_size' in meta:
                        val = meta['object_size']
                        if isinstance(val, (list, tuple)):
                            obj_size = val[0] if len(val) > 0 else None
                        else:
                            obj_size = val
                elif isinstance(meta, (list, tuple)) and len(meta) > 0:
                    # Might be [entry_dict]
                    first = meta[0]
                    if isinstance(first, dict):
                        obj_size = first.get('object_size', None)

                if obj_size in size_stats:
                    size_stats[obj_size]['total'] += 1
                    size_stats[obj_size]['correct'] += int(hit)
            except Exception:
                pass

    acc = correct / total if total > 0 else 0.0
    print(f'Total={total}  Correct={correct}  Acc={acc:.4f}')
    # Print by size
    for k in ['size1', 'size2', 'size3']:
        t = size_stats[k]['total']
        c = size_stats[k]['correct']
        a = (c / t) if t > 0 else 0.0
        print(f'{k}: Total={t}  Correct={c}  Acc={a:.4f}')

if __name__ == '__main__':
    main()


