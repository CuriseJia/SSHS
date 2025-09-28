import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# 复用与 AVENet 相同的基础模型定义
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'comparison', 'IS3'))
from models_lvs import base_models


class CochAV(nn.Module):
    """CochAV: 基于 AVENet 结构的双通道 cochleagram 输入模型。

    - 图像分支: 与 AVENet 相同 (ResNet18)
    - 音频分支: ResNet18, 首层 conv 接收 2 通道输入(左右耳蜗图), 捕获空间音频线索
    - 前向输出: (A, logits, Pos, Neg) 与 AVENet 对齐
    - 支持从 IS3 预训练权重初始化
    """

    def __init__(self, args, pretrained_path=None):
        super(CochAV, self).__init__()

        # Image encoder: 与 AVENet 保持一致
        self.imgnet = base_models.resnet18(modal='vision', pretrained=True)

        # Audio encoder: 基于 AVENet 的 audio resnet18, 但首层支持2通道
        self.audnet = base_models.resnet18(modal='audio')
        # 将第一层从 1 通道(或期望的单通道)扩展到 2 通道
        # 注意：audio模态使用 conv1_a 而不是 conv1
        if hasattr(self.audnet, 'conv1_a'):
            old_conv = self.audnet.conv1_a
            new_conv = nn.Conv2d(
                in_channels=2,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            # 权重初始化: 复制/均值初始化, 让第2通道与第1通道权重一致以稳定迁移
            with torch.no_grad():
                if old_conv.weight.shape[1] == 1:
                    new_conv.weight.data[:, 0:1] = old_conv.weight.data.clone()
                    new_conv.weight.data[:, 1:2] = old_conv.weight.data.clone()
                else:
                    # 若原本不是1通道, 做通道均值
                    mean_weight = old_conv.weight.data.mean(dim=1, keepdim=True)
                    new_conv.weight.data = mean_weight.repeat(1, 2, 1, 1)
                if old_conv.bias is not None:
                    new_conv.bias.data[:] = old_conv.bias.data
            self.audnet.conv1_a = new_conv

        self.m = nn.Sigmoid()
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        self.epsilon = args.epsilon
        self.epsilon2 = args.epsilon2
        self.tau = 0.03
        self.trimap = args.tri_map
        self.Neg = args.Neg

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
        
        # 加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)

    def forward(self, image: torch.Tensor, audio_coch_stereo: torch.Tensor, args, mode: str = 'val'):
        """前向传播。

        Args:
            image: [B, 3, H, W]
            audio_coch_stereo: [B, 2, F, T] 左右耳蜗图双通道
        Returns:
            (A, logits, Pos, Neg) 与 AVENet 一致
        """
        B = image.shape[0]
        mask = (1 - 100 * torch.eye(B, B, device=image.device))

        # Image encoder
        img = self.imgnet(image)
        img = F.normalize(img, dim=1)

        # Audio encoder (stereo cochleagram)
        aud = self.audnet(audio_coch_stereo)
        aud = self.avgpool(aud).view(B, -1)
        aud = F.normalize(aud, dim=1)

        # Join: 与 AVENet 完全一致
        A = torch.einsum('ncqa,nchw->nqa', [img, aud.unsqueeze(2).unsqueeze(3)]).unsqueeze(1)
        A0 = torch.einsum('ncqa,ckhw->nkqa', [img, aud.T.unsqueeze(2).unsqueeze(3)])

        # Trimap
        Pos = self.m((A - self.epsilon) / self.tau)
        if self.trimap:
            Pos2 = self.m((A - self.epsilon2) / self.tau)
            Neg = 1 - Pos2
        else:
            Neg = 1 - Pos

        Pos_all = self.m((A0 - self.epsilon) / self.tau)

        # Positive similarity
        sim1 = (Pos * A).view(*A.shape[:2], -1).sum(-1) / (Pos.view(*Pos.shape[:2], -1).sum(-1))
        # Across negatives
        sim = ((Pos_all * A0).view(*A0.shape[:2], -1).sum(-1) / Pos_all.view(*Pos_all.shape[:2], -1).sum(-1)) * mask
        sim2 = (Neg * A).view(*A.shape[:2], -1).sum(-1) / Neg.view(*Neg.shape[:2], -1).sum(-1)

        if self.Neg:
            logits = torch.cat((sim1, sim, sim2), 1) / 0.07
        else:
            logits = torch.cat((sim1, sim), 1) / 0.07

        return A, logits, Pos, Neg
    
    def _load_pretrained_weights(self, pretrained_path: str):
        """从IS3预训练权重加载模型参数
        
        Args:
            pretrained_path: 预训练权重文件路径 (.tar 或 .pth 文件)
        """
        print(f"正在加载预训练权重: {pretrained_path}")
        
        try:
            # 加载检查点
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # 获取预训练权重字典
            if 'model_state_dict' in checkpoint:
                pretrained_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint
            
            # 获取当前模型状态字典
            model_dict = self.state_dict()
            
            # 过滤掉不匹配的层
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                # 移除 'module.' 前缀（如果存在）
                key = k.replace('module.', '') if k.startswith('module.') else k
                
                # 只加载匹配的层
                if key in model_dict and model_dict[key].shape == v.shape:
                    filtered_dict[key] = v
                else:
                    print(f"跳过不匹配的层: {key} (形状: {v.shape if hasattr(v, 'shape') else 'N/A'})")
            
            # 更新模型字典
            model_dict.update(filtered_dict)
            
            # 加载权重
            self.load_state_dict(model_dict, strict=False)
            
            # print(f"成功加载 {len(filtered_dict)} 个预训练层")
            
            # # 打印加载的层信息
            # print("加载的预训练层:")
            # for key in filtered_dict.keys():
            #     print(f"  - {key}")
                
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
            print("将使用随机初始化的权重")



