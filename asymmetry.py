#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from typing import Tuple, Dict, List
from tqdm import tqdm
from collections import defaultdict
import json

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CochAV 定位偏向性分析')
    parser.add_argument('--config', type=str, default='AudioCOCO/config4.json', help='配置JSON')
    parser.add_argument('--condition', type=str, default='normal', help='无声条件')
    parser.add_argument('--label', type=str, default='no', help='标签')
    parser.add_argument('--image_root', type=str, default='/home/yanhao/coco/val2014/', help='图像根目录')
    parser.add_argument('--coch_root', type=str, default='/data/data0/coch/', help='coch .npy 根目录')
    parser.add_argument('--img_size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=1, help='评测批大小')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader 线程数')
    parser.add_argument('--gpu', type=str, default='3', help='GPU id，例如 0 或 0,1')
    parser.add_argument('--pretrained_path', type=str, required=True, help='cochAV 预训练权重路径 (.pth/.tar)')
    parser.add_argument('--neg', action='store_true', help='启用Neg分支（需与训练一致）')
    parser.add_argument('--tri_map', action='store_true', help='启用Trimap（需与训练一致）')
    parser.add_argument('--epsilon', type=float, default=0.65)
    parser.add_argument('--epsilon2', type=float, default=0.4)
    parser.add_argument('--output_dir', type=str, default='/home/yanhao/SSHS/bias_analysis_results', help='结果输出目录')
    parser.add_argument('--max_samples', type=int, default=500, help='最大评估样本数')
    return parser.parse_args()


def setup_device(gpu: str) -> torch.device:
    """设置设备"""
    if torch.cuda.is_available():
        ids = [int(x) for x in gpu.split(',') if x.strip() != '']
        torch.cuda.set_device(ids[0])
        return torch.device(f'cuda:{ids[0]}')
    return torch.device('cpu')


def distance_to_degree(distance: float, img_size: int = 224) -> float:
    """将距离转换为角度（度数）"""
    return distance / img_size * 25


def calculate_bias_statistics(predictions: List[Dict]) -> Dict:
    """计算偏向性统计信息"""
    # 按object_size分组
    size_groups = defaultdict(list)
    for pred in predictions:
        size_groups[pred['object_size']].append(pred)
    
    results = {}
    
    for size_name, size_predictions in size_groups.items():
        # 按6度以内/以外分组
        x_distance_by_degree = defaultdict(list)
        y_distance_by_degree = defaultdict(list)
        
        for pred in size_predictions:
            x_distance = pred['x_distance']
            y_distance = pred['y_distance']
            
            x_degree = distance_to_degree(x_distance)
            y_degree = distance_to_degree(y_distance)
            
            # 按X距离角度分组：6度以内/以外
            if x_degree < 6.0:
                x_bin_key = "0-6度"
            else:
                x_bin_key = "6度以上"
            x_distance_by_degree[x_bin_key].append(pred)
            
            # 按Y距离角度分组：6度以内/以外
            if y_degree < 6.0:
                y_bin_key = "0-6度"
            else:
                y_bin_key = "6度以上"
            y_distance_by_degree[y_bin_key].append(pred)
        
        results[size_name] = {
            'x_distance_by_degree': x_distance_by_degree,
            'y_distance_by_degree': y_distance_by_degree,
            'total_count': len(size_predictions)
        }
    
    return results


def print_bias_statistics(results: Dict, size_name: str):
    """打印偏向性统计信息"""
    print(f"\n===== {size_name} 定位偏向性分析 =====")
    
    if size_name not in results:
        print(f"未找到 {size_name} 的数据")
        return
    
    size_data = results[size_name]
    x_data = size_data['x_distance_by_degree']
    y_data = size_data['y_distance_by_degree']
    total_count = size_data['total_count']
    
    print(f"总样本数: {total_count}")
    
    # X方向统计
    print("\n--- X方向距离统计 ---")
    print("角度区间\t样本数\t样本比例\t平均X距离\tX距离标准差")
    for bin_key, records in sorted(x_data.items()):
        if records:
            count = len(records)
            ratio = count / total_count if total_count > 0 else 0
            x_distances = [r['x_distance'] for r in records]
            mean_x = np.mean(x_distances)
            std_x = np.std(x_distances)
            print(f"{bin_key}\t{count}\t{ratio:.2%}\t{mean_x:.2f}\t{std_x:.2f}")
    
    # Y方向统计
    print("\n--- Y方向距离统计 ---")
    print("角度区间\t样本数\t样本比例\t平均Y距离\tY距离标准差")
    for bin_key, records in sorted(y_data.items()):
        if records:
            count = len(records)
            ratio = count / total_count if total_count > 0 else 0
            y_distances = [r['y_distance'] for r in records]
            mean_y = np.mean(y_distances)
            std_y = np.std(y_distances)
            print(f"{bin_key}\t{count}\t{ratio:.2%}\t{mean_y:.2f}\t{std_y:.2f}")


def save_detailed_results(results: Dict, output_dir: str):
    """保存详细结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    for size_name, size_data in results.items():
        # 保存X方向数据
        x_data = size_data['x_distance_by_degree']
        for bin_key, records in x_data.items():
            if records:
                safe_bin_key = bin_key.replace('度', 'deg').replace('以上', 'above')
                filename = os.path.join(output_dir, f"{size_name}_x_degree_{safe_bin_key}.txt")
                
                with open(filename, 'w') as f:
                    f.write("peak_x\tpeak_y\tgt_x\tgt_y\tx_distance\ty_distance\tx_degree\ty_degree\timage\taudio\n")
                    for r in records:
                        f.write(f"{r['peak_x']}\t{r['peak_y']}\t{r['gt_x']}\t{r['gt_y']}\t"
                                f"{r['x_distance']:.2f}\t{r['y_distance']:.2f}\t"
                                f"{r['x_degree']:.2f}\t{r['y_degree']:.2f}\t"
                                f"{r['image']}\t{r['audio']}\n")
        
        # 保存Y方向数据
        y_data = size_data['y_distance_by_degree']
        for bin_key, records in y_data.items():
            if records:
                safe_bin_key = bin_key.replace('度', 'deg').replace('以上', 'above')
                filename = os.path.join(output_dir, f"{size_name}_y_degree_{safe_bin_key}.txt")
                
                with open(filename, 'w') as f:
                    f.write("peak_x\tpeak_y\tgt_x\tgt_y\tx_distance\ty_distance\tx_degree\ty_degree\timage\taudio\n")
                    for r in records:
                        f.write(f"{r['peak_x']}\t{r['peak_y']}\t{r['gt_x']}\t{r['gt_y']}\t"
                                f"{r['x_distance']:.2f}\t{r['y_distance']:.2f}\t"
                                f"{r['x_degree']:.2f}\t{r['y_degree']:.2f}\t"
                                f"{r['image']}\t{r['audio']}\n")
    
    # 保存汇总统计
    summary_file = os.path.join(output_dir, "bias_summary.json")
    summary_data = {}
    for size_name, size_data in results.items():
        summary_data[size_name] = {
            'total_count': size_data['total_count'],
            'x_degree_bins': list(size_data['x_distance_by_degree'].keys()),
            'y_degree_bins': list(size_data['y_distance_by_degree'].keys())
        }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)


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

    # 存储所有预测结果
    all_predictions = []

    # 限制评估样本数
    max_samples = min(args.max_samples, len(dataset))
    print(f"将评估 {max_samples} 个样本（总共 {len(dataset)} 个样本）")
    
    with torch.no_grad():
        for idx, (image_t, audio_coch_t, gt, _, _) in enumerate(tqdm(loader, total=max_samples, desc='分析定位偏向性')):
            if idx >= max_samples:
                break
            image_t = image_t.to(device, non_blocking=True)
            audio_coch_t = audio_coch_t.to(device, non_blocking=True)
            
            if args.label == 'noise':
                image_t = torch.rand_like(image_t)
            else:
                image_t = torch.zeros_like(image_t)

            # 前向，拿到 A (特征图)
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
            target_size = args.img_size
            scale_x = target_size / float(w)
            scale_y = target_size / float(h)
            peak_x_224 = int(round((max_x + 0.5) * scale_x))
            peak_y_224 = int(round((max_y + 0.5) * scale_y))

            # 计算GT box中心坐标
            gt_center_x = (xmin + xmax) / 2
            gt_center_y = (ymin + ymax) / 2

            # 计算距离
            x_distance = abs(peak_x_224 - gt_center_x)
            y_distance = abs(peak_y_224 - gt_center_y)
            x_degree = distance_to_degree(x_distance, args.img_size)
            y_degree = distance_to_degree(y_distance, args.img_size)

            # 获取object_size
            try:
                meta = gt['meta']
                obj_size = None
                if isinstance(meta, dict):
                    if 'object_size' in meta:
                        val = meta['object_size']
                        if isinstance(val, (list, tuple)):
                            obj_size = val[0] if len(val) > 0 else 'unknown'
                        else:
                            obj_size = val
                elif isinstance(meta, (list, tuple)) and len(meta) > 0:
                    first = meta[0]
                    if isinstance(first, dict):
                        obj_size = first.get('object_size', 'unknown')
                
                if obj_size is None:
                    obj_size = 'unknown'
            except Exception:
                obj_size = 'unknown'

            # 获取图像和音频文件名
            try:
                image_name = meta.get('image', 'unknown') if isinstance(meta, dict) else 'unknown'
                audio_name = meta.get('audio', 'unknown') if isinstance(meta, dict) else 'unknown'
            except Exception:
                image_name = 'unknown'
                audio_name = 'unknown'

            # 存储预测结果
            prediction = {
                'peak_x': peak_x_224,
                'peak_y': peak_y_224,
                'gt_x': gt_center_x,
                'gt_y': gt_center_y,
                'x_distance': x_distance,
                'y_distance': y_distance,
                'x_degree': x_degree,
                'y_degree': y_degree,
                'object_size': obj_size,
                'image': image_name,
                'audio': audio_name
            }
            all_predictions.append(prediction)

    # 计算偏向性统计
    print(f"总共处理了 {len(all_predictions)} 个样本")
    results = calculate_bias_statistics(all_predictions)

    # 打印统计信息
    for size_name in ['size1', 'size2', 'size3']:
        if size_name in results:
            print_bias_statistics(results, size_name)

    # 保存详细结果
    save_detailed_results(results, args.output_dir)
    print(f"\n详细结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
