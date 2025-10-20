#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CochAV training script - Based on AudioCOCO dataset
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from tqdm import tqdm
from PIL import Image
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be disabled")

from sklearn.metrics import auc

# Import modules
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models import EchoPin_M, EchoPin_S, EchoPin
from AudioCOCO.dataset import create_npy_dataloader, create_audio_coco_dataloader
from AudioCOCO.cochleargram_config import get_config


def get_args():
    parser = argparse.ArgumentParser(description='EchoPin training script')
    
    # data
    parser.add_argument('--train_config', type=str, default='/home/yanhao/SSHS/AudioCOCO/train.json',
                       help='training config file path')
    parser.add_argument('--val_config', type=str, default='/home/yanhao/SSHS/AudioCOCO/train.json',
                       help='validation config file path')
    parser.add_argument('--image_root', type=str, default='/home/yanhao/coco/train2014/',
                       help='image root directory')
    parser.add_argument('--audio_root', type=str, default='/home/yanhao/',
                       help='audio root directory')
    parser.add_argument('--coch_root', type=str, default='/home/yanhao/coch_train/',
                       help='cochleagram .npy root directory')
    
    # model
    parser.add_argument('--coch_config', type=str, default='default',
                       choices=['default', 'speech', 'music', 'high_quality'],
                       help='cochleagram configuration')
    parser.add_argument('--img_size', type=int, default=224,
                       help='image size')
    parser.add_argument('--model_type', type=str, default='EchoPin',
                       choices=['EchoPin', 'EchoPin_M', 'EchoPin_S'],
                       help='model type')
    
    # training
    parser.add_argument('--epochs', type=int, default=10,
                       help='training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='weight decay')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='gradient clipping threshold')
    
    # optimization
    parser.add_argument('--use_amp', action='store_true',
                       help='use mixed precision training')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='learning rate scheduler')
    
    # system
    parser.add_argument('--num_workers', type=int, default=0,
                       help='data loader worker count')
    parser.add_argument('--seed', type=int, default=42,
                       help='random seed')
    parser.add_argument('--distributed', action='store_true',
                       help='use distributed training')
    parser.add_argument('--gpu_ids', type=str, default="1",
                       help='GPU ID, separated by comma, e.g. "0,1,2,3" or "0"')
    parser.add_argument('--force_cpu', action='store_true',
                       help='force using CPU training (even if GPU is available)')
    
    # logging and saving
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='checkpoint save directory')
    parser.add_argument('--experiment_name', type=str, default='cochav_exp',
                       help='experiment name')
    parser.add_argument('--use_wandb', action='store_true',
                       help='use wandb record')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='log interval')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='model save interval')
    
    # EchoPin hyperparameters
    parser.add_argument('--epsilon', type=float, default=0.65,
                       help='positive sample threshold')
    parser.add_argument('--epsilon2', type=float, default=0.4,
                       help='negative sample threshold')
    parser.add_argument('--tri_map', action='store_true',
                       help='use tri-value mask')
    parser.add_argument('--Neg', action='store_true',
                       help='use negative sample')
    
    # pretrained weights
    parser.add_argument('--pretrained_path', type=str, default='/home/yanhao/SSHS/checkpoints/ours_sup_previs.pth.tar',
                       help='IS3 pretrained weights file path (.tar or .pth file)')
    
    # small object optimization
    parser.add_argument('--small_obj_weight', type=float, default=2.0,
                       help='small object weight multiplier')
    parser.add_argument('--detection_lr_mult', type=float, default=5.0,
                       help='检测头学习率倍数，相对于基础学习率')
    
    return parser.parse_args()


class EchoPinTrainer:
    
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.scaler = GradScaler() if args.use_amp else None
        
        # Set random seed
        self._set_seed(args.seed)
        
        # Setup GPU device
        self._setup_gpu()
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._build_dataloaders()
        self._initialize_detection_bias()
        
        # Initialize loss function
        self.criterion = self._build_criterion()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Logging and checkpoints
        self._setup_logging()
        
    def _set_seed(self, seed: int):
        """Set random seed"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _setup_device(self):
        """Setup computing device"""
        if self.args.force_cpu or not torch.cuda.is_available():
            if self.args.force_cpu:
                print("Force using CPU for training")
            else:
                print("CUDA not available, using CPU for training")
            return torch.device('cpu')
        
        # If GPU devices are specified
        if hasattr(self.args, 'gpu_ids') and self.args.gpu_ids:
            if isinstance(self.args.gpu_ids, str):
                gpu_ids = [int(x.strip()) for x in self.args.gpu_ids.split(',')]
            else:
                gpu_ids = self.args.gpu_ids
            
            # Check GPU availability and test GPU health status
            available_gpus = torch.cuda.device_count()
            valid_gpus = []
            
            for gpu_id in gpu_ids:
                if 0 <= gpu_id < available_gpus:
                    try:
                        # Test if GPU can allocate memory normally
                        torch.cuda.set_device(gpu_id)
                        test_tensor = torch.zeros(100, device=f'cuda:{gpu_id}')
                        test_tensor = test_tensor + 1  # Simple operation test
                        del test_tensor
                        torch.cuda.empty_cache()
                        valid_gpus.append(gpu_id)
                        print(f"GPU {gpu_id}: Health check passed")
                    except Exception as e:
                        print(f"GPU {gpu_id}: Health check failed - {e}")
                        print(f"Skipping GPU {gpu_id}")
                else:
                    print(f"GPU {gpu_id}: Index out of range (total {available_gpus} GPUs)")
            
            if not valid_gpus:
                print(f"All specified GPUs {gpu_ids} are unavailable, using all available GPUs")
                valid_gpus = list(range(available_gpus))
            
            self.args.gpu_ids = valid_gpus
            print(f"Finally using GPUs: {valid_gpus}")
            return torch.device(f'cuda:{valid_gpus[0]}')
        else:
            # Use all available GPUs
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                self.args.gpu_ids = list(range(gpu_count))
                print(f"Using all available GPUs: {self.args.gpu_ids}")
            else:
                self.args.gpu_ids = [0]
                print("Using single GPU: 0")
            return torch.device('cuda:0')
    
    def _setup_gpu(self):
        """Setup GPU environment"""
        if self.device.type == 'cuda':
            # Set CUDA device
            torch.cuda.set_device(self.device)

            # Do not modify CUDA_VISIBLE_DEVICES at runtime to avoid mapping conflicts with DataParallel device_ids
            if hasattr(self.args, 'gpu_ids') and len(self.args.gpu_ids) > 1:
                print(f"Multi-GPU training, using device IDs: {self.args.gpu_ids}")
        
    def _build_model(self) -> nn.Module:
        """Build CochAV model"""

        if self.args.model_type == 'EchoPin':
            print(f"Building EchoPin model")
            model = EchoPin(self.args, pretrained_path=getattr(self.args, 'pretrained_path', None))
            model = model.to(self.device)
        elif self.args.model_type == 'EchoPin_M':
            print(f"Building EchoPin_M model")
            model = EchoPin_M(self.args, pretrained_path=getattr(self.args, 'pretrained_path', None))
            model = model.to(self.device)
        elif self.args.model_type == 'EchoPin_S':
            print(f"Building EchoPin_S model")
            model = EchoPin_S(self.args, pretrained_path=getattr(self.args, 'pretrained_path', None))
            model = model.to(self.device)
        else:
            raise ValueError(f"Invalid model type: {self.args.model_type}")

        # Multi-GPU data parallelism
        if self.device.type == 'cuda' and len(self.args.gpu_ids) > 1 and not self.args.distributed:
            # Set CUDA synchronization to avoid multi-GPU deadlock
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(self.args.gpu_ids[0])
            model = nn.DataParallel(model, device_ids=self.args.gpu_ids)
            print(f"Using {len(self.args.gpu_ids)} GPUs for data parallel training: {self.args.gpu_ids}")
            print("Multi-GPU mode: DataLoader multiprocessing disabled to avoid deadlock")
        elif self.device.type == 'cuda' and len(self.args.gpu_ids) == 1:
            print(f"Using single GPU training: GPU {self.args.gpu_ids[0]}")
        else:
            print("Using CPU training")
            
        return model
        
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer, optimized for small object detection"""
        # Separate parameters: pretrained layers use smaller learning rate, detection head uses higher learning rate
        pretrained_params = []
        detection_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'imgnet' in name and param.requires_grad:
                pretrained_params.append(param)
            elif 'det_head' in name and param.requires_grad:
                detection_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': pretrained_params, 'lr': self.args.lr * 0.1},  # Pretrained layers use smaller learning rate
            {'params': detection_params, 'lr': self.args.lr * self.args.detection_lr_mult},  # Detection head uses higher learning rate, beneficial for small objects
            {'params': other_params, 'lr': self.args.lr}
        ], weight_decay=self.args.weight_decay)
        
        return optimizer
        
    def _build_scheduler(self) -> Optional[Any]:
        """Build learning rate scheduler"""
        if self.args.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.epochs, eta_min=self.args.lr * 0.01
            )
        elif self.args.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.args.epochs // 3, gamma=0.1
            )
        elif self.args.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        return None
        
    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Build training and validation data loaders"""
        
        # 根据model_type选择不同的数据加载器
        if self.args.model_type in ['EchoPin_M', 'EchoPin_S']:
            # 使用AudioCocoDataset，直接处理音频文件
            _train_loader, train_dataset = create_audio_coco_dataloader(
                config_json_path=self.args.train_config,
                image_root=self.args.image_root,
                audio_root=self.args.audio_root,
                img_size=self.args.img_size,
                model_type=self.args.model_type,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=True,
                train=True
            )
        else:
            # 使用预生成的cochleagram .npy文件
            _train_loader, train_dataset = create_npy_dataloader(
                config_json_path=self.args.train_config,
                image_root=self.args.image_root,
                coch_root=self.args.coch_root,
                img_size=self.args.img_size,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=True,
                train=True
            )
        
        # Optimize for small objects: calculate sample weights, higher weights for small objects
        self._compute_sample_weights(train_dataset)
        
        def _collate_with_pad(batch):
            # 检查batch格式，AudioCocoDataset和AudioCochDataset返回格式不同
            if len(batch[0]) == 5:
                images, audio_data, gts, neg_images, neg_audio_data = zip(*batch)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch[0])} elements")
            
            images = torch.stack(images, dim=0)
            neg_images = torch.stack(neg_images, dim=0)
            
            # 处理音频数据的时间维度填充
            max_T = max(a.shape[-1] for a in audio_data)
            padded_audio = []
            for a in audio_data:
                pad_T = max_T - a.shape[-1]
                if pad_T > 0:
                    a = F.pad(a, (0, pad_T))
                padded_audio.append(a)
            audio_data = torch.stack(padded_audio, dim=0)
            
            # 处理负样本音频数据的时间维度填充
            max_T_neg = max(a.shape[-1] for a in neg_audio_data)
            max_T_all = max(max_T, max_T_neg)
            if max_T_all != max_T:
                # 需要将正样本填充到新的最大长度
                extra = max_T_all - max_T
                if extra > 0:
                    audio_data = torch.nn.functional.pad(audio_data, (0, extra))
            
            padded_neg_audio = []
            for a in neg_audio_data:
                pad_T = max_T_all - a.shape[-1]
                if pad_T > 0:
                    a = F.pad(a, (0, pad_T))
                padded_neg_audio.append(a)
            neg_audio_data = torch.stack(padded_neg_audio, dim=0)
            
            # 处理GT数据
            bbox = torch.stack([g['bbox_xyxy_224'] for g in gts], dim=0)
            gt_map = torch.stack([g['gt_map_224'] for g in gts], dim=0)
            orig_sizes = [g['orig_size'] for g in gts]
            metas = [g['meta'] for g in gts]
            gt = {
                'bbox_xyxy_224': bbox,
                'gt_map_224': gt_map,
                'orig_size': orig_sizes,
                'meta': metas,
            }
            return images, audio_data, gt, neg_images, neg_audio_data
        
        # Reduce num_workers for multi-GPU to avoid process communication deadlock
        if getattr(self.args, 'distributed', False):
            num_workers = self.args.num_workers
        else:
            num_workers = 0 if len(self.args.gpu_ids) > 1 else self.args.num_workers
        
        self.train_sampler = None
        train_sampler = None
        shuffle_flag = True
        if getattr(self.args, 'distributed', False):
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Distributed training requested but torch.distributed not initialized.")
            self.train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
            train_sampler = self.train_sampler
            shuffle_flag = False
            print("Using DistributedSampler for training")
        else:
            train_sampler = getattr(self, 'weighted_sampler', None)
            if train_sampler is not None:
                shuffle_flag = False
                print("Using weighted sampler for training")
            else:
                shuffle_flag = True
                print("Using random shuffle for training")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            shuffle=shuffle_flag if train_sampler is None else False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_collate_with_pad,
            persistent_workers=False,  # Avoid deadlock caused by persistent worker processes
        )
        
        # Validation set
        if self.args.model_type in ['EchoPin_M', 'EchoPin_S']:
            # 使用AudioCocoDataset，直接处理音频文件
            _val_loader, val_dataset = create_audio_coco_dataloader(
                config_json_path=self.args.val_config,
                image_root=self.args.image_root,
                audio_root=self.args.audio_root,
                img_size=self.args.img_size,
                model_type=self.args.model_type,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
                train=False
            )
        else:
            # 使用预生成的cochleagram .npy文件
            _val_loader, val_dataset = create_npy_dataloader(
                config_json_path=self.args.val_config,
                image_root=self.args.image_root,
                coch_root=self.args.coch_root,
                img_size=self.args.img_size,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
                train=False
            )
        self.val_sampler = None
        val_sampler = None
        if getattr(self.args, 'distributed', False):
            self.val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
            val_sampler = self.val_sampler
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            sampler=val_sampler,
            shuffle=False if val_sampler is None else False,
            num_workers=num_workers,  # Use same worker settings
            pin_memory=True,
            collate_fn=_collate_with_pad,
            persistent_workers=False,
        )
        
        return train_loader, val_loader
    
    def _compute_sample_weights(self, dataset):
        """Calculate sample weights, higher weights for small objects"""
        self.sample_weights = []
        areas = []
        cx_values = []
        cy_values = []
        w_values = []
        h_values = []
        
        if not hasattr(self, '_image_size_cache'):
            self._image_size_cache = {}
        for i in range(len(dataset)):
            entry = dataset.entries[i]
            # Get GT bbox area
            if 'gt_box' in entry:
                bbox = entry['gt_box']
                if isinstance(bbox, list) and len(bbox) == 4:
                    # bbox format is [x, y, w, h]
                    width = max(float(bbox[2]), 1e-3)
                    height = max(float(bbox[3]), 1e-3)
                    area = width * height
                    src_w = entry.get('width') or entry.get('image_width')
                    src_h = entry.get('height') or entry.get('image_height')
                    if src_w is None or src_h is None:
                        img_id = entry.get('image_id')
                        if img_id:
                            if img_id not in self._image_size_cache:
                                img_path = os.path.join(self.args.image_root, img_id)
                                try:
                                    with Image.open(img_path) as img:
                                        self._image_size_cache[img_id] = img.size
                                except Exception:
                                    self._image_size_cache[img_id] = (self.args.img_size, self.args.img_size)
                            src_w, src_h = self._image_size_cache[img_id]
                        else:
                            src_w = src_h = self.args.img_size
                    src_w = max(float(src_w), 1.0)
                    src_h = max(float(src_h), 1.0)
                    x_min = max(float(bbox[0]), 0.0)
                    y_min = max(float(bbox[1]), 0.0)
                    x_max = min(x_min + width, float(src_w))
                    y_max = min(y_min + height, float(src_h))
                    clamped_width = max(x_max - x_min, 1e-3)
                    clamped_height = max(y_max - y_min, 1e-3)
                    area = clamped_width * clamped_height
                    cx_values.append((x_min + 0.5 * clamped_width) / src_w)
                    cy_values.append((y_min + 0.5 * clamped_height) / src_h)
                    w_values.append(clamped_width / src_w)
                    h_values.append(clamped_height / src_h)
                else:
                    area = 1.0  # Default area
            else:
                area = 1.0
            
            areas.append(area)
        
        # Calculate area statistics
        areas = np.array(areas)
        min_area = np.min(areas)
        max_area = np.max(areas)
        
        # Use more stable weight calculation method
        for area in areas:
            # Normalize area to [0, 1]
            if max_area > min_area:
                normalized_area = (area - min_area) / (max_area - min_area)
            else:
                normalized_area = 0.5  # If all areas are the same, use medium weight
            
            # Smaller area has higher weight, use square root function to smooth weight distribution
            weight = 1.0 + self.args.small_obj_weight * np.sqrt(1.0 - normalized_area)
            self.sample_weights.append(weight)
        
        # Create weighted sampler if not using distributed training
        self.sample_weights = torch.tensor(self.sample_weights, dtype=torch.float)
        # Ensure all weights are positive and in reasonable range
        self.sample_weights = torch.clamp(self.sample_weights, min=0.1, max=10.0)
        
        if getattr(self.args, 'distributed', False):
            self.weighted_sampler = None
        else:
            try:
                self.weighted_sampler = torch.utils.data.WeightedRandomSampler(
                    weights=self.sample_weights,
                    num_samples=len(dataset),
                    replacement=True
                )
            except Exception as e:
                print(f"Weighted sampler initialization failed, fallback to random sampling: {e}")
                self.weighted_sampler = None
        if cx_values:
            cx_arr = np.array(cx_values)
            cy_arr = np.array(cy_values)
            w_arr = np.array(w_values)
            h_arr = np.array(h_values)
            self.box_stats = {
                'mean_cx': float(np.mean(cx_arr)),
                'mean_cy': float(np.mean(cy_arr)),
                'median_cx': float(np.median(cx_arr)),
                'median_cy': float(np.median(cy_arr)),
                'mean_w': float(np.mean(w_arr)),
                'mean_h': float(np.mean(h_arr)),
                'median_w': float(np.median(w_arr)),
                'median_h': float(np.median(h_arr)),
            }
        else:
            self.box_stats = None

    def _initialize_detection_bias(self):
        """Initialize detection head bias with dataset statistics if available."""
        module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if not hasattr(module, 'det_head') or not hasattr(self, 'box_stats') or not self.box_stats:
            return

        stats = self.box_stats
        target_vals = [
            stats.get('median_cx', stats['mean_cx']),
            stats.get('median_cy', stats['mean_cy']),
            stats.get('median_w', stats['mean_w']),
            stats.get('median_h', stats['mean_h']),
        ]
        limits = [(0.05, 0.95), (0.05, 0.95), (0.05, 0.8), (0.05, 0.8)]
        target = torch.tensor(
            [
                float(np.clip(val, low, high))
                for val, (low, high) in zip(target_vals, limits)
            ],
            device=module.det_head[-1].bias.device,
            dtype=module.det_head[-1].bias.dtype,
        )
        target = torch.clamp(target, 1e-3, 1 - 1e-3)
        bias_value = torch.logit(target)
        with torch.no_grad():
            module.det_head[-1].bias.copy_(bias_value)
        print(f"[debug] detection bias init to {target.tolist()} (logit={bias_value.tolist()})")
        
    def _build_criterion(self) -> nn.Module:
        """Build loss function"""
        return nn.CrossEntropyLoss()
        
    def _setup_logging(self):
        """Setup logging and wandb"""
        if self.args.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="CochAV-AudioCOCO",
                config=vars(self.args),
                name=f"cochav_{self.args.experiment_name}"
            )
        elif self.args.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available")
            
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        batch_losses = []  # Record loss for each batch
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (images, audio_data, gt, neg_images, neg_audio_data) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            audio_data = audio_data.to(self.device, non_blocking=True)
            neg_images = neg_images.to(self.device, non_blocking=True)
            neg_audio_data = neg_audio_data.to(self.device, non_blocking=True)
            
            # Mixed precision training
            if self.scaler:
                with autocast():
                    A, logits, Pos, Neg, pred_bbox = self.model(images, audio_data, self.args, mode='train')
                    # Negative image pairs
                    _, logits_img_neg, _, _, _ = self.model(neg_images, audio_data, self.args, mode='train')
                    # Negative audio pairs
                    _, logits_aud_neg, _, _, _ = self.model(images, neg_audio_data, self.args, mode='train')
                    loss, loss_img, loss_aud, loss_iou = self._compute_full_loss(logits, logits_img_neg, logits_aud_neg, pred_bbox, gt)
                    
                # Gradient accumulation
                loss = loss / self.args.accumulation_steps
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.args.accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            else:
                A, logits, Pos, Neg, pred_bbox = self.model(images, audio_data, self.args, mode='train')
                _, logits_img_neg, _, _, _ = self.model(neg_images, audio_data, self.args, mode='train')
                _, logits_aud_neg, _, _, _ = self.model(images, neg_audio_data, self.args, mode='train')
                loss, loss_img, loss_aud, loss_iou = self._compute_full_loss(logits, logits_img_neg, logits_aud_neg, pred_bbox, gt)
                
                loss = loss / self.args.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.args.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            
            # Record original total loss (note that loss has been divided by accumulation_steps)
            batch_loss = loss.item() * self.args.accumulation_steps
            total_loss += batch_loss
            batch_losses.append(batch_loss)  # Record current batch loss
            
            # Debug output: print IoU details and A distribution for first 10 batches, especially focus on small objects
            if self.epoch == 0 and batch_idx < 20:
                with torch.no_grad():
                    img_size_dbg = self.args.img_size
                    # Convert center point + width/height to pixel xyxy (consistent with loss)
                    cx = pred_bbox[:, 0]
                    cy = pred_bbox[:, 1]
                    w = pred_bbox[:, 2].clamp(min=0.05)
                    h = pred_bbox[:, 3].clamp(min=0.05)
                    half_w = 0.5 * w * img_size_dbg
                    half_h = 0.5 * h * img_size_dbg
                    cx_pix = cx * img_size_dbg
                    cy_pix = cy * img_size_dbg
                    xmin_dbg = torch.clamp(cx_pix - half_w, 0, img_size_dbg - 1)
                    ymin_dbg = torch.clamp(cy_pix - half_h, 0, img_size_dbg - 1)
                    xmax_dbg = torch.clamp(cx_pix + half_w, 0, img_size_dbg)
                    ymax_dbg = torch.clamp(cy_pix + half_h, 0, img_size_dbg)
                    pred_xyxy_dbg = torch.stack([xmin_dbg, ymin_dbg, xmax_dbg, ymax_dbg], dim=1)
                    gt_xyxy_dbg = gt['bbox_xyxy_224'].to(pred_xyxy_dbg.device).float()
                    iou_vals = self._bbox_iou(pred_xyxy_dbg, gt_xyxy_dbg)
                    
                    # Calculate GT area, identify small objects
                    gt_areas = (gt_xyxy_dbg[:, 2] - gt_xyxy_dbg[:, 0]) * (gt_xyxy_dbg[:, 3] - gt_xyxy_dbg[:, 1])
                    small_obj_mask = gt_areas < (img_size_dbg * img_size_dbg * 0.01)  # Objects with area less than 1% are small objects
                    small_obj_iou = iou_vals[small_obj_mask].mean() if small_obj_mask.any() else torch.tensor(0.0)
                    
                    a_mean = A.mean().item() if isinstance(A, torch.Tensor) else float('nan')
                    a_std = A.std().item() if isinstance(A, torch.Tensor) else float('nan')
                    print(f"[debug][e{self.epoch} b{batch_idx}] iou_mean={iou_vals.mean().item():.3f} "
                          f"small_obj_iou={small_obj_iou.item():.3f} small_obj_count={small_obj_mask.sum().item()} "
                          f"pred0={pred_xyxy_dbg[0].tolist()} gt0={gt_xyxy_dbg[0].tolist()} "
                          f"A_mean={a_mean:.4f} A_std={a_std:.4f}")

            # Update progress bar, show current loss and average loss
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'img': f'{loss_img.item():.3f}',
                'aud': f'{loss_aud.item():.3f}',
                'iou': f'{loss_iou.item():.3f}',
                'iouv': f'{(1.0 - loss_iou.item()):.3f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Logging - record loss changes for each batch
            if self.global_step % self.args.log_interval == 0:
                # Calculate loss statistics for recent batches
                recent_losses = batch_losses[-10:] if len(batch_losses) >= 10 else batch_losses
                loss_std = np.std(recent_losses) if len(recent_losses) > 1 else 0.0
                loss_trend = np.mean(np.diff(recent_losses)) if len(recent_losses) > 1 else 0.0
                
                self._log_metrics({
                    'train/loss': batch_loss,
                    'train/loss_img': loss_img.item(),
                    'train/loss_aud': loss_aud.item(),
                    'train/loss_iou': loss_iou.item(),
                    'train/avg_loss': avg_loss,
                    'train/loss_std': loss_std,
                    'train/loss_trend': loss_trend,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/step': self.global_step,
                    'train/batch': batch_idx
                })
        
        # Calculate loss statistics
        avg_loss = total_loss / num_batches
        loss_std = np.std(batch_losses) if len(batch_losses) > 1 else 0.0
        min_loss = min(batch_losses) if batch_losses else 0.0
        max_loss = max(batch_losses) if batch_losses else 0.0
        
        return {
            'loss': avg_loss,
            'loss_std': loss_std,
            'min_loss': min_loss,
            'max_loss': max_loss,
            'batch_count': len(batch_losses)
        }
        
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_samples = 0
        val_losses = []  # Record loss for each validation batch
        
        with torch.no_grad():
            for batch_idx, (images, audio_data, gt, neg_images, neg_audio_data) in enumerate(tqdm(self.val_loader, desc="Validation")):
                images = images.to(self.device, non_blocking=True)
                audio_data = audio_data.to(self.device, non_blocking=True)
                neg_images = neg_images.to(self.device, non_blocking=True)
                neg_audio_data = neg_audio_data.to(self.device, non_blocking=True)
                
                if self.scaler:
                    with autocast():
                        A, logits, Pos, Neg, pred_bbox = self.model(images, audio_data, self.args, mode='val')
                        _, logits_img_neg, _, _, _ = self.model(neg_images, audio_data, self.args, mode='val')
                        _, logits_aud_neg, _, _, _ = self.model(images, neg_audio_data, self.args, mode='val')
                        loss, loss_img, loss_aud, loss_iou = self._compute_full_loss(logits, logits_img_neg, logits_aud_neg, pred_bbox, gt)
                else:
                    A, logits, Pos, Neg, pred_bbox = self.model(images, audio_data, self.args, mode='val')
                    _, logits_img_neg, _, _, _ = self.model(neg_images, audio_data, self.args, mode='val')
                    _, logits_aud_neg, _, _, _ = self.model(images, neg_audio_data, self.args, mode='val')
                    loss, loss_img, loss_aud, loss_iou = self._compute_full_loss(logits, logits_img_neg, logits_aud_neg, pred_bbox, gt)
                
                batch_loss = loss.item()
                total_loss += batch_loss
                val_losses.append(batch_loss)
                
                # Calculate accuracy (simplified version)
                predictions = torch.argmax(logits, dim=1)
                # This needs to be adjusted according to actual label format
                accuracy = self._compute_accuracy(predictions, gt)
                batch_size = images.size(0)
                total_accuracy += accuracy * batch_size
                num_samples += batch_size
        
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_accuracy / num_samples if num_samples > 0 else 0.0
        loss_std = np.std(val_losses) if len(val_losses) > 1 else 0.0
        min_val_loss = min(val_losses) if val_losses else 0.0
        max_val_loss = max(val_losses) if val_losses else 0.0
        
        return {
            'val_loss': avg_loss, 
            'val_accuracy': avg_accuracy,
            'val_loss_std': loss_std,
            'val_min_loss': min_val_loss,
            'val_max_loss': max_val_loss,
            'val_batch_count': len(val_losses)
        }
        
    def _compute_full_loss(self, logits_pos: torch.Tensor, logits_img_neg: torch.Tensor, logits_aud_neg: torch.Tensor,
                           pred_norm_bbox: torch.Tensor, gt: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine contrastive loss with IoU loss, optimized for small objects.
        - Contrastive loss: push positive sample scores higher than wrong images/wrong audio.
        - IoU loss: based on predicted bbox and ground truth bbox, higher weight for small objects.
        """
        # Contrastive loss: encourage the positive logit (column 0) to stay above all negatives.
        pos_scores = logits_pos[:, 0]
        if logits_pos.size(1) > 1:
            inbatch_neg_scores, _ = torch.max(logits_pos[:, 1:], dim=1)
        else:
            inbatch_neg_scores = torch.zeros_like(pos_scores)
        img_neg_scores = logits_img_neg[:, 0]
        aud_neg_scores = logits_aud_neg[:, 0]

        margin = 0.2
        loss_img_neg = F.relu(margin + img_neg_scores - pos_scores).mean()
        loss_inbatch = F.relu(margin + inbatch_neg_scores - pos_scores).mean()
        loss_img = loss_img_neg + loss_inbatch
        loss_aud = F.relu(margin + aud_neg_scores - pos_scores).mean()

        # IoU loss (interpret prediction as center point + width/height to avoid degenerating to zero area)
        img_size = self.args.img_size
        # Prediction is [cx, cy, w, h], range (0,1)
        cx = pred_norm_bbox[:, 0]
        cy = pred_norm_bbox[:, 1]
        w = pred_norm_bbox[:, 2]
        h = pred_norm_bbox[:, 3]
        # Ensure width and height have lower bound to avoid zero area
        min_frac = 0.05
        w = torch.clamp(w, min=min_frac)
        h = torch.clamp(h, min=min_frac)
        # Convert to pixel coordinates
        half_w = 0.5 * w * img_size
        half_h = 0.5 * h * img_size
        cx_pix = cx * img_size
        cy_pix = cy * img_size
        xmin = torch.clamp(cx_pix - half_w, 0, img_size - 1)
        ymin = torch.clamp(cy_pix - half_h, 0, img_size - 1)
        xmax = torch.clamp(cx_pix + half_w, 0, img_size)
        ymax = torch.clamp(cx_pix + half_h, 0, img_size)
        pred_xyxy = torch.stack([xmin, ymin, xmax, ymax], dim=1)
        gt_xyxy = gt['bbox_xyxy_224'].to(pred_norm_bbox.device).float()
        iou = self._bbox_iou(pred_xyxy, gt_xyxy)
        
        # Calculate GT bbox area for small object weight adjustment
        gt_areas = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
        # Normalize area to [0, 1], smaller area has higher weight
        max_area = img_size * img_size
        normalized_areas = gt_areas / max_area
        # Small object weight: smaller area has higher weight (between 1.0 and 1+small_obj_weight)
        small_obj_weights = 1.0 + self.args.small_obj_weight * (1.0 - normalized_areas)
        
        # Calculate normalized GT centers and sizes for regression guidance
        gt_width = torch.clamp(gt_xyxy[:, 2] - gt_xyxy[:, 0], min=1.0)
        gt_height = torch.clamp(gt_xyxy[:, 3] - gt_xyxy[:, 1], min=1.0)
        gt_cx = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) * 0.5 / img_size
        gt_cy = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) * 0.5 / img_size
        gt_w = gt_width / img_size
        gt_h = gt_height / img_size

        pred_center = torch.stack([cx, cy], dim=1)
        gt_center = torch.stack([gt_cx, gt_cy], dim=1)
        pred_size = torch.stack([w, h], dim=1)
        gt_size = torch.stack([gt_w, gt_h], dim=1)

        center_loss = F.smooth_l1_loss(pred_center, gt_center, reduction='none').sum(dim=1)
        size_loss = F.smooth_l1_loss(pred_size, gt_size, reduction='none').sum(dim=1)

        # Calculate GIoU for more informative gradients when boxes do not overlap
        iou, giou = self._bbox_giou(pred_xyxy, gt_xyxy)
        giou_loss = (1.0 - giou) * small_obj_weights

        # Combine localization losses with weighting for small objects
        loc_reg_loss = (center_loss + size_loss) * small_obj_weights
        loss_iou = giou_loss.mean() + loc_reg_loss.mean()

        # Adjustable weight: balance contrastive and localization terms
        total_loss = loss_img + loss_aud + 2.0 * loss_iou
        return total_loss, loss_img, loss_aud, loss_iou

    @staticmethod
    def _bbox_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU, input [B,4] xyxy."""
        inter_xmin = torch.maximum(box1[:, 0], box2[:, 0])
        inter_ymin = torch.maximum(box1[:, 1], box2[:, 1])
        inter_xmax = torch.minimum(box1[:, 2], box2[:, 2])
        inter_ymax = torch.minimum(box1[:, 3], box2[:, 3])
        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
        inter_area = inter_w * inter_h
        area1 = torch.clamp(box1[:, 2] - box1[:, 0], min=0) * torch.clamp(box1[:, 3] - box1[:, 1], min=0)
        area2 = torch.clamp(box2[:, 2] - box2[:, 0], min=0) * torch.clamp(box2[:, 3] - box2[:, 1], min=0)
        union = area1 + area2 - inter_area + 1e-6
        return inter_area / union
    
    @staticmethod
    def _bbox_giou(box1: torch.Tensor, box2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate IoU and GIoU for xyxy boxes."""
        inter_xmin = torch.maximum(box1[:, 0], box2[:, 0])
        inter_ymin = torch.maximum(box1[:, 1], box2[:, 1])
        inter_xmax = torch.minimum(box1[:, 2], box2[:, 2])
        inter_ymax = torch.minimum(box1[:, 3], box2[:, 3])
        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
        inter_area = inter_w * inter_h

        area1 = torch.clamp(box1[:, 2] - box1[:, 0], min=0) * torch.clamp(box1[:, 3] - box1[:, 1], min=0)
        area2 = torch.clamp(box2[:, 2] - box2[:, 0], min=0) * torch.clamp(box2[:, 3] - box2[:, 1], min=0)
        union = area1 + area2 - inter_area + 1e-6
        iou = inter_area / union

        c_xmin = torch.minimum(box1[:, 0], box2[:, 0])
        c_ymin = torch.minimum(box1[:, 1], box2[:, 1])
        c_xmax = torch.maximum(box1[:, 2], box2[:, 2])
        c_ymax = torch.maximum(box1[:, 3], box2[:, 3])
        c_w = torch.clamp(c_xmax - c_xmin, min=0)
        c_h = torch.clamp(c_ymax - c_ymin, min=0)
        c_area = c_w * c_h + 1e-6

        giou = iou - (c_area - union) / c_area
        return iou, giou
        
    def _compute_accuracy(self, predictions: torch.Tensor, gt: Dict[str, Any]) -> float:
        """计算准确率"""
        # 简化版本，实际需要根据任务调整
        batch_size = predictions.size(0)
        correct = (predictions == 0).sum().item()  # 假设正样本标签为0
        return correct / batch_size
        
    def _log_metrics(self, metrics: Dict[str, float]):
        """记录指标"""
        if self.args.use_wandb and WANDB_AVAILABLE:
            wandb.log(metrics, step=self.global_step)
            
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'args': self.args
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.args.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到: {best_path}")
            
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"从 {checkpoint_path} 加载检查点，epoch: {self.epoch}")
        
    def train(self):
        """主训练循环"""
        print("开始训练CochAV模型...")
        print(f"设备: {self.device}")
        print(f"批次大小: {self.args.batch_size}")
        print(f"学习率: {self.args.lr}")
        print(f"总epochs: {self.args.epochs}")
        
        for epoch in range(self.epoch, self.args.epochs):
            self.epoch = epoch
            start_time = time.time()
            
            if getattr(self.args, 'distributed', False) and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # 记录指标
            epoch_time = time.time() - start_time
            metrics = {
                **train_metrics,
                **val_metrics,
                'epoch': epoch,
                'epoch_time': epoch_time
            }
            
            self._log_metrics(metrics)
            
            # 打印结果 - 包含详细的loss统计
            print(f"Epoch {epoch}: "
                  f"train_loss={train_metrics['loss']:.4f}±{train_metrics['loss_std']:.4f} "
                  f"(min:{train_metrics['min_loss']:.4f}, max:{train_metrics['max_loss']:.4f}), "
                  f"val_loss={val_metrics['val_loss']:.4f}±{val_metrics['val_loss_std']:.4f} "
                  f"(min:{val_metrics['val_min_loss']:.4f}, max:{val_metrics['val_max_loss']:.4f}), "
                  f"val_acc={val_metrics['val_accuracy']:.4f}, "
                  f"batches={train_metrics['batch_count']}/{val_metrics['val_batch_count']}, "
                  f"time={epoch_time:.2f}s")
            
            # 保存检查点
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                
            if epoch % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(is_best)
                
        print("训练完成！")


def main():
    """主函数"""
    args = get_args()
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 创建训练器
    trainer = EchoPinTrainer(args)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
