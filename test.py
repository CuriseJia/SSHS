#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Tuple
from tqdm import tqdm

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CochAV heatmap 命中率评测')
    parser.add_argument('--config', type=str, default='AudioCOCO/config6.json', help='配置JSON')
    parser.add_argument('--condition', type=str, default='normal', help='无声条件')
    parser.add_argument('--label', type=str, default='no', help='标签')
    parser.add_argument('--image_root', type=str, default='/home/yanhao/coco/val2014/', help='图像根目录')
    parser.add_argument('--coch_root', type=str, default='/data/data0/coch/', help='coch .npy 根目录')
    parser.add_argument('--img_size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=32, help='评测批大小')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader 线程数')
    parser.add_argument('--gpu', type=str, default='3', help='GPU id，例如 0 或 0,1')
    parser.add_argument('--pretrained_path', type=str, required=True, help='cochAV 预训练权重路径 (.pth/.tar)')
    parser.add_argument('--neg', action='store_true', help='启用Neg分支（需与训练一致）')
    parser.add_argument('--tri_map', action='store_true', help='启用Trimap（需与训练一致）')
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

    # 延迟导入，避免不必要依赖
    from AudioCOCO.dataset import create_npy_dataloader
    from models.CochAV import CochAV

    # 适配训练脚本中的参数形状
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
    # 分size统计
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

            # 最大值坐标
            max_idx = torch.argmax(heatmap)
            h, w = heatmap.shape
            max_y = (max_idx // w).item()
            max_x = (max_idx % w).item()

            # GT bbox (已为 224 尺度的 xyxy，long)
            bbox_xyxy = gt['bbox_xyxy_224'][0].to(device).long()
            xmin, ymin, xmax, ymax = [int(v.item()) for v in bbox_xyxy]

            # heatmap 尺度 -> 224 尺度坐标对齐
            # ResNet18 输出 14x14 特征图，需要映射到 224x224 图像
            target_size = args.img_size  # 224
            scale_x = target_size / float(w)  # 224/14 = 16
            scale_y = target_size / float(h)  # 224/14 = 16
            peak_x_224 = int(round((max_x + 0.5) * scale_x))
            peak_y_224 = int(round((max_y + 0.5) * scale_y))

            # 命中判断：峰值是否落入 gt_box
            hit = (xmin <= peak_x_224 <= xmax) and (ymin <= peak_y_224 <= ymax)
            correct += int(hit)
            total += 1

            # 从 config 条目获取 object_size 分组（兼容不同collate形态）
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
                    # 可能是 [entry_dict]
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
    # 分 size 打印
    for k in ['size1', 'size2', 'size3']:
        t = size_stats[k]['total']
        c = size_stats[k]['correct']
        a = (c / t) if t > 0 else 0.0
        print(f'{k}: Total={t}  Correct={c}  Acc={a:.4f}')

if __name__ == '__main__':
    main()


