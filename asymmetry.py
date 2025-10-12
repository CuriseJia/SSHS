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
    parser = argparse.ArgumentParser(description='CochAV localization bias analysis')
    parser.add_argument('--config', type=str, default='AudioCOCO/config4.json', help='Configuration JSON')
    parser.add_argument('--condition', type=str, default='normal', help='Silent condition')
    parser.add_argument('--label', type=str, default='no', help='Label')
    parser.add_argument('--image_root', type=str, default='/home/yanhao/coco/val2014/', help='Image root directory')
    parser.add_argument('--coch_root', type=str, default='/data/data0/coch/', help='coch .npy root directory')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--batch_size', type=int, default=1, help='Evaluation batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader thread count')
    parser.add_argument('--gpu', type=str, default='3', help='GPU id, e.g. 0 or 0,1')
    parser.add_argument('--pretrained_path', type=str, required=True, help='cochAV pretrained weight path (.pth/.tar)')
    parser.add_argument('--neg', action='store_true', help='Enable Neg branch (must be consistent with training)')
    parser.add_argument('--tri_map', action='store_true', help='Enable Trimap (must be consistent with training)')
    parser.add_argument('--epsilon', type=float, default=0.65)
    parser.add_argument('--epsilon2', type=float, default=0.4)
    parser.add_argument('--output_dir', type=str, default='/home/yanhao/SSHS/bias_analysis_results', help='Result output directory')
    parser.add_argument('--max_samples', type=int, default=500, help='Maximum evaluation samples')
    return parser.parse_args()


def setup_device(gpu: str) -> torch.device:
    """Setup device"""
    if torch.cuda.is_available():
        ids = [int(x) for x in gpu.split(',') if x.strip() != '']
        torch.cuda.set_device(ids[0])
        return torch.device(f'cuda:{ids[0]}')
    return torch.device('cpu')


def distance_to_degree(distance: float, img_size: int = 224) -> float:
    """Convert distance to angle (degrees)"""
    return distance / img_size * 25


def calculate_bias_statistics(predictions: List[Dict]) -> Dict:
    """Calculate bias statistics"""
    # Group by object_size
    size_groups = defaultdict(list)
    for pred in predictions:
        size_groups[pred['object_size']].append(pred)
    
    results = {}
    
    for size_name, size_predictions in size_groups.items():
        # Group by within/outside 6 degrees
        x_distance_by_degree = defaultdict(list)
        y_distance_by_degree = defaultdict(list)
        
        for pred in size_predictions:
            x_distance = pred['x_distance']
            y_distance = pred['y_distance']
            
            x_degree = distance_to_degree(x_distance)
            y_degree = distance_to_degree(y_distance)
            
            # Group X distance by angle: within/outside 6 degrees
            if x_degree < 6.0:
                x_bin_key = "0-6deg"
            else:
                x_bin_key = "6deg+above"
            x_distance_by_degree[x_bin_key].append(pred)
            
            # Group Y distance by angle: within/outside 6 degrees
            if y_degree < 6.0:
                y_bin_key = "0-6deg"
            else:
                y_bin_key = "6deg+above"
            y_distance_by_degree[y_bin_key].append(pred)
        
        results[size_name] = {
            'x_distance_by_degree': x_distance_by_degree,
            'y_distance_by_degree': y_distance_by_degree,
            'total_count': len(size_predictions)
        }
    
    return results


def print_bias_statistics(results: Dict, size_name: str):
    """Print bias statistics"""
    print(f"\n===== {size_name} Localization Bias Analysis =====")
    
    if size_name not in results:
        print(f"No data found for {size_name}")
        return
    
    size_data = results[size_name]
    x_data = size_data['x_distance_by_degree']
    y_data = size_data['y_distance_by_degree']
    total_count = size_data['total_count']
    
    print(f"Total samples: {total_count}")
    
    # X direction statistics
    print("\n--- X Direction Distance Statistics ---")
    print("Angle Range\tSample Count\tSample Ratio\tAvg X Distance\tX Distance Std")
    for bin_key, records in sorted(x_data.items()):
        if records:
            count = len(records)
            ratio = count / total_count if total_count > 0 else 0
            x_distances = [r['x_distance'] for r in records]
            mean_x = np.mean(x_distances)
            std_x = np.std(x_distances)
            print(f"{bin_key}\t{count}\t{ratio:.2%}\t{mean_x:.2f}\t{std_x:.2f}")
    
    # Y direction statistics
    print("\n--- Y Direction Distance Statistics ---")
    print("Angle Range\tSample Count\tSample Ratio\tAvg Y Distance\tY Distance Std")
    for bin_key, records in sorted(y_data.items()):
        if records:
            count = len(records)
            ratio = count / total_count if total_count > 0 else 0
            y_distances = [r['y_distance'] for r in records]
            mean_y = np.mean(y_distances)
            std_y = np.std(y_distances)
            print(f"{bin_key}\t{count}\t{ratio:.2%}\t{mean_y:.2f}\t{std_y:.2f}")


def save_detailed_results(results: Dict, output_dir: str):
    """Save detailed results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for size_name, size_data in results.items():
        # Save X direction data
        x_data = size_data['x_distance_by_degree']
        for bin_key, records in x_data.items():
            if records:
                safe_bin_key = bin_key.replace('deg', 'deg').replace('+above', 'above')
                filename = os.path.join(output_dir, f"{size_name}_x_degree_{safe_bin_key}.txt")
                
                with open(filename, 'w') as f:
                    f.write("peak_x\tpeak_y\tgt_x\tgt_y\tx_distance\ty_distance\tx_degree\ty_degree\timage\taudio\n")
                    for r in records:
                        f.write(f"{r['peak_x']}\t{r['peak_y']}\t{r['gt_x']}\t{r['gt_y']}\t"
                                f"{r['x_distance']:.2f}\t{r['y_distance']:.2f}\t"
                                f"{r['x_degree']:.2f}\t{r['y_degree']:.2f}\t"
                                f"{r['image']}\t{r['audio']}\n")
        
        # Save Y direction data
        y_data = size_data['y_distance_by_degree']
        for bin_key, records in y_data.items():
            if records:
                safe_bin_key = bin_key.replace('deg', 'deg').replace('+above', 'above')
                filename = os.path.join(output_dir, f"{size_name}_y_degree_{safe_bin_key}.txt")
                
                with open(filename, 'w') as f:
                    f.write("peak_x\tpeak_y\tgt_x\tgt_y\tx_distance\ty_distance\tx_degree\ty_degree\timage\taudio\n")
                    for r in records:
                        f.write(f"{r['peak_x']}\t{r['peak_y']}\t{r['gt_x']}\t{r['gt_y']}\t"
                                f"{r['x_distance']:.2f}\t{r['y_distance']:.2f}\t"
                                f"{r['x_degree']:.2f}\t{r['y_degree']:.2f}\t"
                                f"{r['image']}\t{r['audio']}\n")
    
    # Save summary statistics
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

    # Store all prediction results
    all_predictions = []

    # Limit evaluation sample count
    max_samples = min(args.max_samples, len(dataset))
    print(f"Will evaluate {max_samples} samples (total {len(dataset)} samples)")
    
    with torch.no_grad():
        for idx, (image_t, audio_coch_t, gt, _, _) in enumerate(tqdm(loader, total=max_samples, desc='Analyzing localization bias')):
            if idx >= max_samples:
                break
            image_t = image_t.to(device, non_blocking=True)
            audio_coch_t = audio_coch_t.to(device, non_blocking=True)
            
            if args.label == 'noise':
                image_t = torch.rand_like(image_t)
            else:
                image_t = torch.zeros_like(image_t)

            # Forward pass, get A (feature map)
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
            target_size = args.img_size
            scale_x = target_size / float(w)
            scale_y = target_size / float(h)
            peak_x_224 = int(round((max_x + 0.5) * scale_x))
            peak_y_224 = int(round((max_y + 0.5) * scale_y))

            # Calculate GT box center coordinates
            gt_center_x = (xmin + xmax) / 2
            gt_center_y = (ymin + ymax) / 2

            # Calculate distance
            x_distance = abs(peak_x_224 - gt_center_x)
            y_distance = abs(peak_y_224 - gt_center_y)
            x_degree = distance_to_degree(x_distance, args.img_size)
            y_degree = distance_to_degree(y_distance, args.img_size)

            # Get object_size
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

            # Get image and audio file names
            try:
                image_name = meta.get('image', 'unknown') if isinstance(meta, dict) else 'unknown'
                audio_name = meta.get('audio', 'unknown') if isinstance(meta, dict) else 'unknown'
            except Exception:
                image_name = 'unknown'
                audio_name = 'unknown'

            # Store prediction results
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

    # Calculate bias statistics
    print(f"Total processed {len(all_predictions)} samples")
    results = calculate_bias_statistics(all_predictions)

    # Print statistics
    for size_name in ['size1', 'size2', 'size3']:
        if size_name in results:
            print_bias_statistics(results, size_name)

    # Save detailed results
    save_detailed_results(results, args.output_dir)
    print(f"\nDetailed results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
