#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Dict, List, Tuple, Optional, Set

import torch
import numpy as np
from tqdm import tqdm
import easydict
from comparison.IS3.model_lvs import AVENet

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CochAV heatmap vacc evaluation (mask hit rate)')
    parser.add_argument('--model', type=str, default='IS3', help='EchoPin or IS3')
    parser.add_argument('--label', type=str, default='silent', help='silent or noise')
    parser.add_argument('--config', type=str, default='/home/yanhao/SSHS/AudioCOCO/finalConfig/config4.json', help='Configuration JSON')
    parser.add_argument('--image_root', type=str, default='/home/yanhao/coco/val2014/', help='Image root directory')
    parser.add_argument('--coch_root', type=str, default='/home/yanhao/', help='coch .npy root directory')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--batch_size', type=int, default=4, help='Evaluation batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader thread count')
    parser.add_argument('--gpu', type=str, default='2', help='GPU id, e.g. 0 or 0,1')
    parser.add_argument('--pretrained_path', type=str, default='/home/yanhao/SSHS/checkpoints/ours_sup_previs.pth.tar', help='cochAV pretrained weight path (.pth/.tar)')
    parser.add_argument('--instances_json', type=str, default='/home/yanhao/SSHS/AudioCOCO/instances_val2014.json', help='COCO instances json path')
    parser.add_argument('--category_txt', type=str, default='/home/yanhao/SSHS/AudioCOCO/category.txt', help='Category name list (one per line)')
    return parser.parse_args()


def setup_device(gpu: str) -> torch.device:
    if torch.cuda.is_available():
        ids = [int(x) for x in gpu.split(',') if x.strip() != '']
        torch.cuda.set_device(ids[0])
        return torch.device(f'cuda:{ids[0]}')
    return torch.device('cpu')


def load_categories(category_txt_path: str) -> Set[str]:
    names: Set[str] = set()
    with open(category_txt_path, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                names.add(name)
    return names


def build_coco_indices(instances_json_path: str, target_cat_names: Set[str]):
    """Build fast indices to get per-image masks for all categories (for V-acc calculation).

    Returns:
      image_file_to_id: file_name -> image_id
      image_id_to_size: image_id -> (height, width)
      image_id_to_anns: image_id -> [ann dict for all categories]
      cat_id_to_name: category_id -> name
    """
    import json
    with open(instances_json_path, 'r') as f:
        coco = json.load(f)

    cat_id_to_name: Dict[int, str] = {c['id']: c['name'] for c in coco['categories']}

    image_file_to_id: Dict[str, int] = {}
    image_id_to_size: Dict[int, Tuple[int, int]] = {}
    for im in coco['images']:
        image_file_to_id[im['file_name']] = im['id']
        image_id_to_size[im['id']] = (im['height'], im['width'])

    # 为了计算 V-acc，我们需要所有类别的注释，而不仅仅是目标类别
    image_id_to_anns: Dict[int, List[dict]] = {}
    for ann in coco['annotations']:
        image_id_to_anns.setdefault(ann['image_id'], []).append(ann)

    return image_file_to_id, image_id_to_size, image_id_to_anns, cat_id_to_name


def build_union_mask_for_image(
    anns: List[dict],
    img_h: int,
    img_w: int,
    out_size: int = 224,
) -> np.ndarray:
    """Create a union binary mask (H=W=out_size) for given annotations."""
    try:
        from pycocotools import mask as maskUtils
    except Exception as e:
        raise RuntimeError('pycocotools is required for vacc evaluation. Please install it: pip install pycocotools') from e

    if len(anns) == 0:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    union = np.zeros((img_h, img_w), dtype=np.uint8)
    for ann in anns:
        # 统一使用 annToRLE，可同时处理 polygon 与 RLE（含 counts 为 list 的非压缩格式）
        try:
            rle = maskUtils.annToRLE(ann)
        except Exception:
            # 退化处理：尝试直接从 segmentation 构造
            segm = ann.get('segmentation', None)
            if segm is None:
                continue
            if isinstance(segm, list):
                rles = maskUtils.frPyObjects(segm, img_h, img_w)
                rle = maskUtils.merge(rles)
            elif isinstance(segm, dict):
                # 若为未压缩RLE，确保为有效RLE对象
                rle = maskUtils.frPyObjects(segm, img_h, img_w)
                if isinstance(rle, list):
                    rle = maskUtils.merge(rle)
        m = maskUtils.decode(rle)  # [H, W]
        if m.ndim == 3:
            m = m[:, :, 0]
        union = np.logical_or(union, m > 0)

    union = union.astype(np.uint8)  # 0/1
    if (img_h, img_w) == (out_size, out_size):
        return union

    # resize to out_size with nearest
    import cv2
    mask_resized = cv2.resize(union, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    mask_resized = (mask_resized > 0).astype(np.uint8)
    return mask_resized


def main() -> None:
    args = parse_args()
    device = setup_device(args.gpu)

    # Lazy imports
    from AudioCOCO.dataset import create_npy_dataloader, create_audio_coco_dataloader
    from models.EchoPin import EchoPin

    # Load COCO indices and category set
    target_cat_names = load_categories(args.category_txt)
    image_file_to_id, image_id_to_size, image_id_to_anns, _ = build_coco_indices(
        args.instances_json, target_cat_names
    )

    # Data
    if args.model == 'EchoPin':
        loader, dataset = create_npy_dataloader(
            config_json_path=args.config,
            image_root=args.image_root,
            coch_root=args.coch_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            train=False,
        )
    else:
        loader, dataset = create_audio_coco_dataloader(
            config_json_path=args.config,
            image_root=args.image_root,
            audio_root=args.coch_root,
            img_size=args.img_size,
            model_type='EchoPin_M',
        )

    # Model

    lvs_args = easydict.EasyDict({
        "epsilon" : 0.65,
        "epsilon2" : 0.4,
        'tri_map' : True,
        'Neg' : True,
        'tau' : 0.03,
        })

    class EvalArgs:
        def __init__(self, ns: argparse.Namespace):
            self.epsilon = getattr(ns, 'epsilon', 0.65)
            self.epsilon2 = getattr(ns, 'epsilon2', 0.4)
            self.tri_map = getattr(ns, 'tri_map', False)
            self.Neg = getattr(ns, 'neg', False)
            self.img_size = ns.img_size
            self.pretrained_path = ns.pretrained_path
            self.gpu_ids = [int(x) for x in ns.gpu.split(',') if x.strip() != ''] or [0]

    eval_args = EvalArgs(args)

    if args.model == 'EchoPin':
        model = EchoPin(eval_args, pretrained_path=eval_args.pretrained_path).to(device)
    elif args.model == 'IS3':
        model = AVENet(lvs_args).to(device).eval()
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.eval()

    correct = 0
    total = 0
    size_stats = {
        'size1': {'correct': 0, 'total': 0},
        'size2': {'correct': 0, 'total': 0},
        'size3': {'correct': 0, 'total': 0},
    }

    with torch.no_grad():
        for image_t, audio_coch_t, gt, _, _ in tqdm(loader, total=len(dataset), desc='Evaluating VACC'):
            image_t = image_t.to(device, non_blocking=True)
            audio_coch_t = audio_coch_t.to(device, non_blocking=True)
            if args.label == 'silent':
                audio_coch_t = torch.zeros_like(audio_coch_t)
            elif args.label == 'noise':
                audio_coch_t = torch.randn_like(audio_coch_t)
            # 使用 feature map 计算 heatmap，类似 test_temp.py 的逻辑
            image_feature = model.imgnet(image_t)
            audio_feature = model.audnet(audio_coch_t)
            
            batch_size = image_t.size(0)
            metas = gt['meta']

            for b in range(batch_size):
                # 提取单个样本的 feature
                F_img_flat = image_feature[b].view(512, -1)  # [512, H*W]
                F_aud_flat = audio_feature[b].view(512, -1)  # [512, H*W]

                # 归一化
                F_img_norm = F_img_flat / F_img_flat.norm(dim=0, keepdim=True)
                F_aud_norm = F_aud_flat / F_aud_flat.norm(dim=0, keepdim=True)

                # 计算相似度矩阵
                S = torch.matmul(F_img_norm.transpose(0, 1), F_aud_norm)

                # 获取每个空间位置的最大相似度分数
                Image_Scores_flat = torch.max(S, dim=1).values
                heatmap_low = Image_Scores_flat.view(14, 14)

                # 插值到 224x224
                heatmap_tensor = heatmap_low.unsqueeze(0).unsqueeze(0)
                heatmap_high = torch.nn.functional.interpolate(
                    heatmap_tensor, 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                )
                heatmap_final = heatmap_high.squeeze()

                # 找到最高点坐标
                max_flat_index = torch.argmax(heatmap_final)
                h, w = heatmap_final.shape
                max_y = (max_flat_index // w).item()
                max_x = (max_flat_index % w).item()

                # entry for this sample
                entry = metas
                if isinstance(entry, (list, tuple)) and len(entry) > b:
                    entry = entry[b]
                # if still dict, use as-is; else set to empty dict
                if not isinstance(entry, dict):
                    entry = {}

                img_rel = entry.get('image', entry.get('image_id'))
                # 兼容 list/tuple/dict 等多种形式
                if isinstance(img_rel, (list, tuple)):
                    img_rel = img_rel[0] if len(img_rel) > 0 else ''
                elif isinstance(img_rel, dict):
                    img_rel = img_rel.get('file_name', '')
                if not isinstance(img_rel, str):
                    img_rel = str(img_rel)
                file_name = os.path.basename(img_rel)
                image_id = image_file_to_id.get(file_name, None)

                hit = False
                if image_id is not None:
                    # 仿照 variants.py 的逻辑，读取 COCO 中对应图像的所有 gt_box
                    anns_all = image_id_to_anns.get(image_id, [])
                    img_h, img_w = image_id_to_size.get(image_id, (args.img_size, args.img_size))
                    union_mask_224 = build_union_mask_for_image(anns_all, img_h, img_w, out_size=args.img_size)
                    if 0 <= max_y < args.img_size and 0 <= max_x < args.img_size:
                        hit = bool(union_mask_224[max_y, max_x] > 0)

                correct += int(hit)
                total += 1

                # size bucket for this sample
                obj_size: Optional[str] = None
                try:
                    meta_b = metas[b] if isinstance(metas, (list, tuple)) and len(metas) > b else metas
                    if isinstance(meta_b, dict):
                        val = meta_b.get('object_size', None)
                        if isinstance(val, (list, tuple)):
                            obj_size = val[0] if len(val) > 0 else None
                        else:
                            obj_size = val
                except Exception:
                    obj_size = None

                if obj_size in size_stats:
                    size_stats[obj_size]['total'] += 1
                    size_stats[obj_size]['correct'] += int(hit)

    acc = (correct / total) if total > 0 else 0.0
    print(f'Total={total}  Correct={correct}  VACC={acc:.4f}')
    for k in ['size1', 'size2', 'size3']:
        t = size_stats[k]['total']
        c = size_stats[k]['correct']
        a = (c / t) if t > 0 else 0.0
        print(f'{k}: Total={t}  Correct={c}  VACC={a:.4f}')


if __name__ == '__main__':
    main()


