import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, Optional

try:
    # 期望与 AVENet 相同的依赖来源
    from comparison.IS3.models_lvs import base_models
except Exception:
    base_models = None

from SSHS.AudioCOCO.dataset import AudioCocoProcessor


class StereoToMono1DCNN(nn.Module):
    """将立体声 [B,2,L] 提取空间差异并映射到单通道波形 [B,L]。

    设计目标:
    - 保持时序长度 L 不变 (padding='same' 等效)
    - 多尺度卷积融合左右声道信息
    - 末端输出1通道, 通过 Tanh 限幅到 [-1,1]
    """

    def __init__(self, in_channels: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, stereo_wav: torch.Tensor) -> torch.Tensor:
        """stereo_wav: [B,2,L] -> mono_wav: [B,L]"""
        mono = self.net(stereo_wav)  # [B,1,L]
        return mono.squeeze(1)


class CochAV(nn.Module):
    """基于 AVENet 的结构, 将音频端替换为: Stereo-1D-CNN -> Processor(cochleagram) -> Audio-ResNet。

    输入:
    - image: [B,3,H,W] (如 224x224)
    - stereo_wav: [B,2,L] (原始立体声波形, 采样率 sample_rate)

    输出与 AVENet 一致:
    - A, logits, Pos, Neg
    """

    def __init__(
        self,
        coch_config: Optional[Dict[str, Any]] = None,
        sample_rate: int = 16000,
        epsilon: float = 0.65,
        epsilon2: float = 0.4,
        tri_map: bool = True,
        use_neg: bool = True,
        tau: float = 0.03,
        vision_pretrained: bool = True,
    ) -> None:
        super().__init__()

        # 图像/音频编码器与 AVENet 对齐
        if base_models is None:
            raise ImportError("无法导入 models_lvs.base_models，请确保运行路径与 AVENet 一致并可导入 'models_lvs'")

        self.imgnet = base_models.resnet18(modal='vision', pretrained=vision_pretrained)
        self.audnet = base_models.resnet18(modal='audio')
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        # 立体声 -> 单声道波形
        self.s2m = StereoToMono1DCNN(in_channels=2, hidden=64)

        # cochleagram 处理器 (numpy域), 前向中将做 CPU 往返, 用于推理/评估
        self.sample_rate = sample_rate
        self.processor = AudioCocoProcessor(**(coch_config or {}))

        # AVENet 对比学习参数
        self.epsilon = epsilon
        self.epsilon2 = epsilon2
        self.tau = tau
        self.trimap = tri_map
        self.Neg = use_neg

        # 初始化与 AVENet 对齐
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, image: torch.Tensor, stereo_wav: torch.Tensor) -> tuple:
        """
        image: [B,3,H,W]
        stereo_wav: [B,2,L]
        返回: A, logits, Pos, Neg (与 AVENet 一致)
        """
        device = image.device
        batch_size = image.shape[0]

        # 图像编码
        img = self.imgnet(image)
        img = F.normalize(img, dim=1)

        # 立体声 -> 单通道波形 (torch域)
        mono_wav = self.s2m(stereo_wav)  # [B,L]

        # 调用 numpy 域 processor 生成 cochleagram (逐样本处理)
        coch_list = []
        for b in range(batch_size):
            wav_np = mono_wav[b].detach().cpu().float().numpy()
            coch_np = self.processor(wav_np, sr=self.sample_rate)  # [F,T] numpy
            coch_t = torch.from_numpy(coch_np).unsqueeze(0).unsqueeze(0).float()  # [1,1,F,T]
            coch_list.append(coch_t)
        audio_coch = torch.cat(coch_list, dim=0).to(device)  # [B,1,F,T]

        # 音频编码 (复用 AVENet)
        aud_feat = self.audnet(audio_coch)
        aud_vec = self.avgpool(aud_feat).view(batch_size, -1)
        aud_vec = F.normalize(aud_vec, dim=1)

        # 融合 (与 AVENet 相同)
        A = torch.einsum('ncqa,nchw->nqa', [img, aud_vec.unsqueeze(2).unsqueeze(3)]).unsqueeze(1)
        A0 = torch.einsum('ncqa,ckhw->nkqa', [img, aud_vec.T.unsqueeze(2).unsqueeze(3)])

        Pos = self.sigmoid((A - self.epsilon) / self.tau)
        if self.trimap:
            Pos2 = self.sigmoid((A - self.epsilon2) / self.tau)
            Neg = 1 - Pos2
        else:
            Neg = 1 - Pos

        Pos_all = self.sigmoid((A0 - self.epsilon) / self.tau)

        sim1 = (Pos * A).view(*A.shape[:2], -1).sum(-1) / (Pos.view(*Pos.shape[:2], -1).sum(-1))
        mask = (1 - 100 * torch.eye(batch_size, batch_size, device=device))
        sim = ((Pos_all * A0).view(*A0.shape[:2], -1).sum(-1) / Pos_all.view(*Pos_all.shape[:2], -1).sum(-1)) * mask
        sim2 = (Neg * A).view(*A.shape[:2], -1).sum(-1) / Neg.view(*Neg.shape[:2], -1).sum(-1)

        if self.Neg:
            logits = torch.cat((sim1, sim, sim2), 1) / 0.07
        else:
            logits = torch.cat((sim1, sim), 1) / 0.07

        return A, logits, Pos, Neg


__all__ = [
    'CochAV',
]


