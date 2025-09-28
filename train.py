#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CochAV 训练脚本 - AudioCOCO数据集
包含性能优化策略：混合精度、梯度累积、学习率调度、数据并行等
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

# 导入我们的模块
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.CochAV import CochAV
from AudioCOCO.dataset import create_dataloader
from AudioCOCO.cochleargram_config import get_config


class CochAVTrainer:
    """CochAV训练器，包含完整的训练流程和优化策略"""
    
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.scaler = GradScaler() if args.use_amp else None
        
        # 设置随机种子
        self._set_seed(args.seed)
        
        # 设置GPU设备
        self._setup_gpu()
        
        # 初始化模型
        self.model = self._build_model()
        
        # 初始化优化器和调度器
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # 初始化数据加载器
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # 初始化损失函数
        self.criterion = self._build_criterion()
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 日志和检查点
        self._setup_logging()
        
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _setup_device(self):
        """设置计算设备"""
        if self.args.force_cpu or not torch.cuda.is_available():
            if self.args.force_cpu:
                print("强制使用CPU训练")
            else:
                print("CUDA不可用，使用CPU训练")
            return torch.device('cpu')
        
        # 如果指定了GPU设备
        if hasattr(self.args, 'gpu_ids') and self.args.gpu_ids:
            if isinstance(self.args.gpu_ids, str):
                gpu_ids = [int(x.strip()) for x in self.args.gpu_ids.split(',')]
            else:
                gpu_ids = self.args.gpu_ids
            
            # 检查GPU是否可用
            available_gpus = torch.cuda.device_count()
            valid_gpus = [gpu_id for gpu_id in gpu_ids if 0 <= gpu_id < available_gpus]
            
            if not valid_gpus:
                print(f"指定的GPU {gpu_ids} 不可用，使用所有可用GPU")
                valid_gpus = list(range(available_gpus))
            
            self.args.gpu_ids = valid_gpus
            print(f"使用GPU: {valid_gpus}")
            return torch.device(f'cuda:{valid_gpus[0]}')
        else:
            # 使用所有可用GPU
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                self.args.gpu_ids = list(range(gpu_count))
                print(f"使用所有可用GPU: {self.args.gpu_ids}")
            else:
                self.args.gpu_ids = [0]
                print("使用单GPU: 0")
            return torch.device('cuda:0')
    
    def _setup_gpu(self):
        """设置GPU环境"""
        if self.device.type == 'cuda':
            # 设置CUDA设备
            torch.cuda.set_device(self.device)
            
            # 设置环境变量（用于多GPU训练）
            if hasattr(self.args, 'gpu_ids') and len(self.args.gpu_ids) > 1:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.args.gpu_ids))
                print(f"设置CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        
    def _build_model(self) -> nn.Module:
        """构建CochAV模型"""
        model = CochAV(self.args, pretrained_path=getattr(self.args, 'pretrained_path', None))
        model = model.to(self.device)
        
        # 多GPU数据并行
        if self.device.type == 'cuda' and len(self.args.gpu_ids) > 1 and not self.args.distributed:
            model = nn.DataParallel(model, device_ids=self.args.gpu_ids)
            print(f"使用 {len(self.args.gpu_ids)} 个GPU进行数据并行训练: {self.args.gpu_ids}")
        elif self.device.type == 'cuda' and len(self.args.gpu_ids) == 1:
            print(f"使用单GPU训练: GPU {self.args.gpu_ids[0]}")
        else:
            print("使用CPU训练")
            
        return model
        
    def _build_optimizer(self) -> optim.Optimizer:
        """构建优化器"""
        # 分离参数：预训练层使用较小学习率
        pretrained_params = []
        new_params = []
        
        for name, param in self.model.named_parameters():
            if 'imgnet' in name and param.requires_grad:
                pretrained_params.append(param)
            else:
                new_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': pretrained_params, 'lr': self.args.lr * 0.1},  # 预训练层使用较小学习率
            {'params': new_params, 'lr': self.args.lr}
        ], weight_decay=self.args.weight_decay)
        
        return optimizer
        
    def _build_scheduler(self) -> Optional[Any]:
        """构建学习率调度器"""
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
        """构建训练和验证数据加载器"""
        # 获取cochleagram配置
        coch_config = get_config(self.args.coch_config)
        
        # 训练集
        train_loader, _ = create_dataloader(
            config_json_path=self.args.train_config,
            image_root=self.args.image_root,
            audio_root=self.args.audio_root,
            coch_config=coch_config,
            img_size=self.args.img_size,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            train=True
        )
        
        # 验证集
        val_loader, _ = create_dataloader(
            config_json_path=self.args.val_config,
            image_root=self.args.image_root,
            audio_root=self.args.audio_root,
            coch_config=coch_config,
            img_size=self.args.img_size,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            train=False
        )
        
        return train_loader, val_loader
        
    def _build_criterion(self) -> nn.Module:
        """构建损失函数"""
        return nn.CrossEntropyLoss()
        
    def _setup_logging(self):
        """设置日志和wandb"""
        if self.args.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="CochAV-AudioCOCO",
                config=vars(self.args),
                name=f"cochav_{self.args.experiment_name}"
            )
        elif self.args.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available")
            
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        batch_losses = []  # 记录每个batch的loss
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (images, audio_coch, gt) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            audio_coch = audio_coch.to(self.device, non_blocking=True)
            
            # 混合精度训练
            if self.scaler:
                with autocast():
                    A, logits, Pos, Neg = self.model(images, audio_coch, self.args, mode='train')
                    loss = self._compute_loss(logits, gt)
                    
                # 梯度累积
                loss = loss / self.args.accumulation_steps
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.args.accumulation_steps == 0:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            else:
                A, logits, Pos, Neg = self.model(images, audio_coch, self.args, mode='train')
                loss = self._compute_loss(logits, gt)
                
                loss = loss / self.args.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.args.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            
            batch_loss = loss.item() * self.args.accumulation_steps
            total_loss += batch_loss
            batch_losses.append(batch_loss)  # 记录当前batch的loss
            
            # 更新进度条，显示当前loss和平均loss
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # 日志记录 - 每个batch都记录loss变化
            if self.global_step % self.args.log_interval == 0:
                # 计算最近几个batch的loss统计
                recent_losses = batch_losses[-10:] if len(batch_losses) >= 10 else batch_losses
                loss_std = np.std(recent_losses) if len(recent_losses) > 1 else 0.0
                loss_trend = np.mean(np.diff(recent_losses)) if len(recent_losses) > 1 else 0.0
                
                self._log_metrics({
                    'train/loss': batch_loss,
                    'train/avg_loss': avg_loss,
                    'train/loss_std': loss_std,
                    'train/loss_trend': loss_trend,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/step': self.global_step,
                    'train/batch': batch_idx
                })
        
        # 计算loss统计信息
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
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_samples = 0
        val_losses = []  # 记录每个验证batch的loss
        
        with torch.no_grad():
            for batch_idx, (images, audio_coch, gt) in enumerate(tqdm(self.val_loader, desc="Validation")):
                images = images.to(self.device, non_blocking=True)
                audio_coch = audio_coch.to(self.device, non_blocking=True)
                
                if self.scaler:
                    with autocast():
                        A, logits, Pos, Neg = self.model(images, audio_coch, self.args, mode='val')
                        loss = self._compute_loss(logits, gt)
                else:
                    A, logits, Pos, Neg = self.model(images, audio_coch, self.args, mode='val')
                    loss = self._compute_loss(logits, gt)
                
                batch_loss = loss.item()
                total_loss += batch_loss
                val_losses.append(batch_loss)
                
                # 计算准确率（简化版本）
                predictions = torch.argmax(logits, dim=1)
                # 这里需要根据实际的标签格式调整
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
        
    def _compute_loss(self, logits: torch.Tensor, gt: Dict[str, Any]) -> torch.Tensor:
        """计算损失"""
        # 这里需要根据CochAV的输出格式调整损失计算
        # 暂时使用简单的分类损失
        batch_size = logits.size(0)
        # 创建伪标签（正样本为0，负样本为1）
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        return self.criterion(logits, labels)
        
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
    parser.add_argument('--image_root', type=str, default='/home/yanhao/AudioCOCO/images/single_object/',
                       help='图像根目录')
    parser.add_argument('--audio_root', type=str, default='/home/yanhao/AudioCOCO/audios/',
                       help='音频根目录')
    
    # 模型相关
    parser.add_argument('--coch_config', type=str, default='default',
                       choices=['default', 'speech', 'music', 'high_quality'],
                       help='cochleagram配置')
    parser.add_argument('--img_size', type=int, default=224,
                       help='图像尺寸')
    
    # 训练相关
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--accumulation_steps', type=int, default=1,
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
    parser.add_argument('--gpu_ids', type=str, default="3",
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
