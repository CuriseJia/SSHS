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
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from tqdm import tqdm
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

from models.CochAV import CochAV
from AudioCOCO.dataset import create_npy_dataloader
from AudioCOCO.cochleargram_config import get_config


class CochAVTrainer:
    
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
        model = CochAV(self.args, pretrained_path=getattr(self.args, 'pretrained_path', None))
        model = model.to(self.device)
        
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
        # Get cochleagram configuration
        coch_config = get_config(self.args.coch_config)
        
        # Training set (using pre-generated .npy cochleagram)
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
            images, cochs, gts, neg_images, neg_cochs = zip(*batch)
            images = torch.stack(images, dim=0)
            max_T = max(c.shape[-1] for c in cochs)
            padded = []
            for c in cochs:
                pad_T = max_T - c.shape[-1]
                if pad_T > 0:
                    c = F.pad(c, (0, pad_T))
                padded.append(c)
            cochs = torch.stack(padded, dim=0)
            # Negative sample coch padding
            max_T_neg = max(c.shape[-1] for c in neg_cochs)
            max_T_all = max(max_T, max_T_neg)
            if max_T_all != max_T:
                # Need to pad positive samples to new maximum length
                extra = max_T_all - max_T
                if extra > 0:
                    cochs = torch.nn.functional.pad(cochs, (0, extra))
            padded_neg = []
            for c in neg_cochs:
                pad_T = max_T_all - c.shape[-1]
                if pad_T > 0:
                    c = F.pad(c, (0, pad_T))
                padded_neg.append(c)
            neg_cochs = torch.stack(padded_neg, dim=0)
            neg_images = torch.stack(neg_images, dim=0)
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
            return images, cochs, gt, neg_images, neg_cochs
        
        # Reduce num_workers for multi-GPU to avoid process communication deadlock
        num_workers = 0 if len(self.args.gpu_ids) > 1 else self.args.num_workers
        
        # Try to use weighted sampler, fallback to normal random sampling if failed
        try:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                sampler=self.sampler,  # Use weighted sampler, higher sampling probability for small objects
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=_collate_with_pad,
                persistent_workers=False,  # Avoid deadlock caused by persistent worker processes
            )
            print("Using weighted sampler for training")
        except Exception as e:
            print(f"Weighted sampler failed, fallback to normal random sampling: {e}")
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,  # Use normal random sampling
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=_collate_with_pad,
                persistent_workers=False,
            )
        
        # Validation set (using pre-generated .npy cochleagram)
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
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
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
        
        for i in range(len(dataset)):
            entry = dataset.entries[i]
            # Get GT bbox area
            if 'gt_box' in entry:
                bbox = entry['gt_box']
                if isinstance(bbox, list) and len(bbox) == 4:
                    # bbox format is [x, y, w, h]
                    area = max(bbox[2] * bbox[3], 1.0)  # Ensure area is at least 1
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
        
        # Create weighted sampler
        self.sample_weights = torch.tensor(self.sample_weights, dtype=torch.float)
        # Ensure all weights are positive and in reasonable range
        self.sample_weights = torch.clamp(self.sample_weights, min=0.1, max=10.0)
        
        self.sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(dataset),
            replacement=True
        )
        
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
        
        for batch_idx, (images, audio_coch, gt, neg_images, neg_coch) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            audio_coch = audio_coch.to(self.device, non_blocking=True)
            neg_images = neg_images.to(self.device, non_blocking=True)
            neg_coch = neg_coch.to(self.device, non_blocking=True)
            
            # Mixed precision training
            if self.scaler:
                with autocast():
                    A, logits, Pos, Neg, pred_bbox = self.model(images, audio_coch, self.args, mode='train')
                    # Negative image pairs
                    _, logits_img_neg, _, _, _ = self.model(neg_images, audio_coch, self.args, mode='train')
                    # Negative audio pairs
                    _, logits_aud_neg, _, _, _ = self.model(images, neg_coch, self.args, mode='train')
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
                A, logits, Pos, Neg, pred_bbox = self.model(images, audio_coch, self.args, mode='train')
                _, logits_img_neg, _, _, _ = self.model(neg_images, audio_coch, self.args, mode='train')
                _, logits_aud_neg, _, _, _ = self.model(images, neg_coch, self.args, mode='train')
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
            if self.epoch == 0 and batch_idx < 10:
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
            for batch_idx, (images, audio_coch, gt, neg_images, neg_coch) in enumerate(tqdm(self.val_loader, desc="Validation")):
                images = images.to(self.device, non_blocking=True)
                audio_coch = audio_coch.to(self.device, non_blocking=True)
                neg_images = neg_images.to(self.device, non_blocking=True)
                neg_coch = neg_coch.to(self.device, non_blocking=True)
                
                if self.scaler:
                    with autocast():
                        A, logits, Pos, Neg, pred_bbox = self.model(images, audio_coch, self.args, mode='val')
                        _, logits_img_neg, _, _, _ = self.model(neg_images, audio_coch, self.args, mode='val')
                        _, logits_aud_neg, _, _, _ = self.model(images, neg_coch, self.args, mode='val')
                        loss, loss_img, loss_aud, loss_iou = self._compute_full_loss(logits, logits_img_neg, logits_aud_neg, pred_bbox, gt)
                else:
                    A, logits, Pos, Neg, pred_bbox = self.model(images, audio_coch, self.args, mode='val')
                    _, logits_img_neg, _, _, _ = self.model(neg_images, audio_coch, self.args, mode='val')
                    _, logits_aud_neg, _, _, _ = self.model(images, neg_coch, self.args, mode='val')
                    loss, loss_img, loss_aud, loss_iou = self._compute_full_loss(logits, logits_img_neg, logits_aud_neg, pred_bbox, gt)
                
                batch_loss = loss.item()
                total_loss += batch_loss
                val_losses.append(batch_loss)
                
                # Calculate accuracy (simplified version)
                predictions = torch.argmax(logits, dim=1)
                # This needs to be adjusted according to actual label format
                accuracy = self._compute_accuracy(predictions, gt)
                total_accuracy += accuracy
                num_samples += images.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_accuracy / num_samples
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
        # Use smooth contrastive loss: gentle margin loss + gradient clipping
        sim_pos = logits_pos.mean(dim=1)
        sim_img_neg = logits_img_neg.mean(dim=1)
        sim_aud_neg = logits_aud_neg.mean(dim=1)
        
        # Smooth margin loss to avoid gradient explosion
        margin = 0.5
        loss_img_raw = torch.clamp(margin - sim_pos + sim_img_neg, min=0)
        loss_aud_raw = torch.clamp(margin - sim_pos + sim_aud_neg, min=0)
        
        # Use squared loss smoothing and limit maximum value
        loss_img = torch.clamp(loss_img_raw.pow(2), max=4.0).mean()
        loss_aud = torch.clamp(loss_aud_raw.pow(2), max=4.0).mean()

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
        
        # Weighted IoU loss
        weighted_iou_loss = (1.0 - iou) * small_obj_weights
        loss_iou = weighted_iou_loss.mean()

        # Adjustable weight: increase IoU loss weight to promote localization learning, higher weight for small objects
        total_loss = loss_img + loss_aud + 5.0 * loss_iou  # IoU loss weight 5x
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


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CochAV训练脚本')
    
    # 数据相关
    parser.add_argument('--train_config', type=str, default='AudioCOCO/config1.json',
                       help='训练集配置文件路径')
    parser.add_argument('--val_config', type=str, default='AudioCOCO/config1.json',
                       help='验证集配置文件路径')
    parser.add_argument('--image_root', type=str, default='/home/yanhao/coco/val2014/',
                       help='图像根目录')
    parser.add_argument('--coch_root', type=str, default='/home/yanhao/SSHS/AudioCOCO/coch/',
                       help='cochleagram .npy 根目录')
    
    # 模型相关
    parser.add_argument('--coch_config', type=str, default='default',
                       choices=['default', 'speech', 'music', 'high_quality'],
                       help='cochleagram配置')
    parser.add_argument('--img_size', type=int, default=224,
                       help='图像尺寸')
    
    # 训练相关
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--accumulation_steps', type=int, default=8,
                       help='梯度累积步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='梯度裁剪阈值')
    
    # 优化策略
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度训练')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='学习率调度器')
    
    # 系统相关
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--distributed', action='store_true',
                       help='使用分布式训练')
    parser.add_argument('--gpu_ids', type=str, default="2",
                       help='指定使用的GPU ID，用逗号分隔，如 "0,1,2,3" 或 "0"')
    parser.add_argument('--force_cpu', action='store_true',
                       help='强制使用CPU训练（即使有GPU可用）')
    
    # 日志和保存
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--experiment_name', type=str, default='cochav_exp',
                       help='实验名称')
    parser.add_argument('--use_wandb', action='store_true',
                       help='使用wandb记录')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='日志记录间隔')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='模型保存间隔')
    
    # CochAV特定参数
    parser.add_argument('--epsilon', type=float, default=0.65,
                       help='正样本阈值')
    parser.add_argument('--epsilon2', type=float, default=0.4,
                       help='负样本阈值')
    parser.add_argument('--tri_map', action='store_true',
                       help='使用三值掩码')
    parser.add_argument('--Neg', action='store_true',
                       help='使用负样本')
    
    # 预训练权重
    parser.add_argument('--pretrained_path', type=str, default='/home/yanhao/SSHS/checkpoints/ours_sup_previs.pth.tar',
                       help='IS3预训练权重文件路径 (.tar 或 .pth 文件)')
    
    # 小物体优化参数
    parser.add_argument('--small_obj_weight', type=float, default=3.0,
                       help='小物体权重倍数，越大越关注小物体')
    parser.add_argument('--detection_lr_mult', type=float, default=2.0,
                       help='检测头学习率倍数，相对于基础学习率')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = get_args()
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 创建训练器
    trainer = CochAVTrainer(args)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
