import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .data_preprocess import CochleagramPreprocessor
import soundfile as sf


class AudioCocoProcessor:
    """将单/双通道音频波形或wav路径转换为 cochleagram 的处理器。

    - 若输入为路径: 自动加载；若为立体声，最后一维为2通道
    - 若输入为数组: 支持单通道或多通道，需提供采样率
    - 返回: np.ndarray, 单声道形状为 [F, T]；立体声为 [F, T, 2]
    """

    def __init__(self, **coch_config: Any) -> None:
        self.preprocessor = CochleagramPreprocessor(**coch_config)

    def __call__(self, audio: Union[str, np.ndarray], sr: Optional[int] = None) -> np.ndarray:
        if isinstance(audio, str):
            signal, file_sr = self.preprocessor.load_audio(audio)
            return self.preprocessor.generate_cochleagram(signal, file_sr)
        else:
            assert sr is not None, "当传入numpy波形时必须提供采样率sr"
            signal = audio
            # 若为多通道且包含左右声道，交由预处理器分别转换
            if signal.ndim > 1 and signal.shape[1] >= 2:
                return self.preprocessor.generate_cochleagram(signal, sr)
            # 否则按单通道处理
            if signal.ndim > 1:
                signal = np.mean(signal, axis=1)
            return self.preprocessor.generate_cochleagram(signal, sr)


def _build_img_transform(img_size: int = 224, train: bool = False) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.1), Image.BICUBIC),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def _bbox_xywh_to_xyxy(b: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    return x, y, x + w, y + h


def _resize_bbox(
    bbox_xywh: List[float],
    src_wh: Tuple[int, int],
    dst_hw: Tuple[int, int],
) -> List[int]:
    """将原图坐标的 xywh bbox 映射到 dst 尺度(如224x224)。返回整数 xyxy。"""
    src_w, src_h = src_wh
    dst_h, dst_w = dst_hw
    scale_x = dst_w / max(1e-6, src_w)
    scale_y = dst_h / max(1e-6, src_h)
    x, y, w, h = bbox_xywh
    xmin = int(round(x * scale_x))
    ymin = int(round(y * scale_y))
    xmax = int(round((x + w) * scale_x))
    ymax = int(round((y + h) * scale_y))
    xmin = max(0, min(dst_w - 1, xmin))
    ymin = max(0, min(dst_h - 1, ymin))
    xmax = max(0, min(dst_w, xmax))
    ymax = max(0, min(dst_h, ymax))
    return [xmin, ymin, xmax, ymax]


def _bbox_to_gt_map(bbox_xyxy: List[int], size: int = 224) -> np.ndarray:
    gt = np.zeros((size, size), dtype=np.float32)
    xmin, ymin, xmax, ymax = bbox_xyxy
    if xmax > xmin and ymax > ymin:
        gt[ymin:ymax, xmin:xmax] = 1.0
    return gt


def _maybe_denormalize_xywh(bbox_xywh: List[float], src_wh: Tuple[int, int]) -> List[float]:
    """若检测到 bbox 为归一化(0-1)坐标，则按源分辨率还原为像素坐标。"""
    x, y, w, h = bbox_xywh
    src_w, src_h = src_wh
    # 简单启发式：若宽高均<=1.0，视为归一化
    if max(w, h) <= 1.0 and max(x, y) <= 1.0:
        return [x * src_w, y * src_h, w * src_w, h * src_h]
    return bbox_xywh

class AudioCocoDataset(Dataset):
    """

    每个样本返回:
    - image: FloatTensor [3, H, W] (默认224)
    - audio_coch: FloatTensor [2, F, T] (立体声；若单声道则复制)
    - gt: Dict 包含
        - bbox_xyxy_224: Tensor[int] [4]
        - gt_map_224: FloatTensor [224, 224]
        - orig_size: (img_h, img_w)
        - meta: 原始条目字典(可选)
    """

    def __init__(
        self,
        config_json_path: str,
        image_root: str,
        audio_root: str,
        processor: AudioCocoProcessor,
        img_size: int = 224,
        source_wh: Tuple[int, int] = (1920, 1080),
        coch_target_ft: Optional[Tuple[int, int]] = None,
        train: bool = False,
    ) -> None:
        super().__init__()
        self.image_root = image_root
        self.audio_root = audio_root
        self.processor = processor
        self.img_size = img_size
        self.source_wh = source_wh
        self.coch_target_ft = coch_target_ft
        self.transform = _build_img_transform(img_size, train)

        with open(config_json_path, 'r') as f:
            self.entries: List[Dict[str, Any]] = json.load(f)

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve_paths(self, entry: Dict[str, Any]) -> Tuple[str, str]:
        img_path = os.path.join(self.image_root, entry['image_id'])
        aud_path = os.path.join(self.audio_root, entry['audio'])
        return img_path, aud_path

    def _load_image(self, path: str) -> Tuple[Tensor, Tuple[int, int]]:
        img = Image.open(path).convert('RGB')
        orig_w, orig_h = img.size
        img_t = self.transform(img)  # [3, H, W]
        return img_t, (orig_h, orig_w)

    def _load_audio_coch(self, path: str) -> Tensor:
        # 使用 soundfile 读取以保留声道, 并确保所有音频长度均为10秒
        samples, sr = sf.read(path)
        total_target = sr * 5  # 保留前5秒以便裁出2-5秒窗口
        if samples.ndim == 1:
            # 单声道
            mono = samples
            if len(mono) < total_target:
                mono = np.pad(mono, (0, total_target - len(mono)), mode='constant', constant_values=0)
            else:
                mono = mono[:total_target]
            # 统一从2s到5s取三秒
            start = int(2.0 * sr)
            end = int(5.0 * sr)
            mono = mono[start:end]
            left = mono
            right = mono
        else:
            # 立体声或多声道，取前两个声道
            left_all = samples[:, 0]
            right_all = samples[:, 1]
            # 对齐到前5秒
            if len(left_all) < total_target:
                left_all = np.pad(left_all, (0, total_target - len(left_all)), mode='constant', constant_values=0)
            else:
                left_all = left_all[:total_target]
            if len(right_all) < total_target:
                right_all = np.pad(right_all, (0, total_target - len(right_all)), mode='constant', constant_values=0)
            else:
                right_all = right_all[:total_target]
            # 统一2-5秒窗口
            start = int(2.0 * sr)
            end = int(5.0 * sr)
            left = left_all[start:end]
            right = right_all[start:end]
        # 若多声道，提取左右；若单声道，复制为双通道
        if samples.ndim == 2 and samples.shape[1] >= 2:
            left = samples[:, 0]
            right = samples[:, 1]
        else:
            mono = samples if samples.ndim == 1 else samples[:, 0]
            left = mono
            right = mono
        
        # 确保左右声道长度一致
        min_len = min(len(left), len(right))
        left = left[:min_len]
        right = right[:min_len]
        # print(f'left shape: {left.shape}, right shape: {right.shape}')
        # 分别生成左右 cochleagram（处理器内部可选整流+0.3幂）
        coch_L = self.processor(left, sr=sr)  # [F, T]
        coch_R = self.processor(right, sr=sr)  # [F, T]
        # print(f'coch_L shape: {coch_L.shape}, coch_R shape: {coch_R.shape}')
        coch = np.stack([coch_L, coch_R], axis=0)  # [2, F, T]
        coch_t = torch.from_numpy(coch).float()
        # 可选缩放到目标尺寸 [2, F_tgt, T_tgt]
        if self.coch_target_ft is not None:
            F_tgt, T_tgt = self.coch_target_ft
            coch_t = torch.nn.functional.interpolate(
                coch_t.unsqueeze(0), size=(F_tgt, T_tgt), mode='bilinear', align_corners=False
            ).squeeze(0)
        return coch_t

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        entry = self.entries[idx]
        img_path, aud_path = self._resolve_paths(entry)

        image_t, (orig_h, orig_w) = self._load_image(img_path)
        audio_coch_t = self._load_audio_coch(aud_path)

        bbox_xywh: List[float] = entry['gt_box']
        # 源分辨率优先从条目中获取（例如 'width'/'height' 或 'image_width'/'image_height'），否则使用默认 source_wh
        src_w = entry.get('width', entry.get('image_width', self.source_wh[0]))
        src_h = entry.get('height', entry.get('image_height', self.source_wh[1]))
        bbox_xywh = _maybe_denormalize_xywh(bbox_xywh, (src_w, src_h))
        bbox_xyxy_224 = _resize_bbox(bbox_xywh, (src_w, src_h), (self.img_size, self.img_size))
        gt_map_224 = _bbox_to_gt_map(bbox_xyxy_224, size=self.img_size)

        gt: Dict[str, Any] = {
            'bbox_xyxy_224': torch.tensor(bbox_xyxy_224, dtype=torch.long),
            'gt_map_224': torch.from_numpy(gt_map_224).float(),
            'orig_size': (orig_h, orig_w),
            'meta': entry,
        }

        return image_t, audio_coch_t, gt


class AudioCocoNPYDataset(Dataset):
    """基于配置文件读取图像与已预生成的cochleagram(.npy)的Dataset。

    约定：
    - 配置JSON为列表，每个元素为字典，至少包含：
        - 'output': coch文件的基名（不含扩展名或含，均可），本数据集将读取对应的 .npy 文件
        - 'image' 或 'image_id': 图像相对路径
        - 'gt_box' 或 'bbox' (xywh): 目标框，单位为原图像素坐标
    - 图片路径: image_root / <image or image_id>
    - coch路径: coch_root / (<output> 去掉扩展名).npy

    每个样本返回:
    - image: FloatTensor [3, H, W]
    - audio_coch: FloatTensor [2, F, T]
    - gt: 同 AudioCocoDataset
    """

    def __init__(
        self,
        config_json_path: str,
        image_root: str,
        coch_root: str,
        img_size: int = 224,
        source_wh: Tuple[int, int] = (1920, 1080),
        coch_target_ft: Optional[Tuple[int, int]] = None,
        train: bool = False,
    ) -> None:
        super().__init__()
        self.image_root = image_root
        self.coch_root = coch_root
        self.img_size = img_size
        self.source_wh = source_wh
        self.coch_target_ft = coch_target_ft
        self.transform = _build_img_transform(img_size, train)

        with open(config_json_path, 'r') as f:
            self.entries: List[Dict[str, Any]] = json.load(f)
        # 根据类别建立索引，便于采样负样本
        self.category_key = 'category' if (len(self.entries) > 0 and 'category' in self.entries[0]) else None
        self.cat_to_indices: Dict[Any, List[int]] = {}
        if self.category_key is not None:
            for idx, e in enumerate(self.entries):
                cat = e[self.category_key]
                self.cat_to_indices.setdefault(cat, []).append(idx)

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve_img_path(self, entry: Dict[str, Any]) -> str:
        img_key = 'image' if 'image' in entry else 'image_id'
        return os.path.join(self.image_root, entry[img_key])

    def _resolve_coch_path(self, entry: Dict[str, Any]) -> str:
        output_name = entry['output']
        base = os.path.splitext(output_name)[0]
        return os.path.join(self.coch_root, base + '.npy')

    def _load_image(self, path: str) -> Tuple[Tensor, Tuple[int, int]]:
        img = Image.open(path).convert('RGB')
        orig_w, orig_h = img.size
        img_t = self.transform(img)
        return img_t, (orig_h, orig_w)

    def _load_coch_npy(self, path: str) -> Tensor:
        arr = np.load(path)
        # 统一为 [2, F, T]
        if arr.ndim == 2:
            # [F, T] -> 复制为双通道
            arr = np.stack([arr, arr], axis=0)
        elif arr.ndim == 3:
            # 可能是 [F, T, 2] 或 [2, F, T]
            if arr.shape[-1] == 2:
                arr = np.transpose(arr, (2, 0, 1))
            elif arr.shape[0] == 2:
                # 已是 [2, F, T]
                pass
            else:
                raise ValueError(f"无法识别的coch形状: {arr.shape}, 期望 [F,T], [F,T,2] 或 [2,F,T]")
        else:
            raise ValueError(f"无法识别的coch形状: {arr.shape}")
        coch_t = torch.from_numpy(arr).float()
        if self.coch_target_ft is not None:
            F_tgt, T_tgt = self.coch_target_ft
            coch_t = torch.nn.functional.interpolate(
                coch_t.unsqueeze(0), size=(F_tgt, T_tgt), mode='bilinear', align_corners=False
            ).squeeze(0)
        return coch_t

    def _get_bbox_xywh(self, entry: Dict[str, Any]) -> List[float]:
        if 'gt_box' in entry:
            return entry['gt_box']
        if 'bbox' in entry:
            return entry['bbox']
        raise KeyError("配置项缺少 'gt_box' 或 'bbox' 字段用于监督")

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Dict[str, Any], Tensor, Tensor]:
        entry = self.entries[idx]
        img_path = self._resolve_img_path(entry)
        coch_path = self._resolve_coch_path(entry)

        image_t, (orig_h, orig_w) = self._load_image(img_path)
        audio_coch_t = self._load_coch_npy(coch_path)

        bbox_xywh = self._get_bbox_xywh(entry)
        src_w = entry.get('width', entry.get('image_width', self.source_wh[0]))
        src_h = entry.get('height', entry.get('image_height', self.source_wh[1]))
        bbox_xywh = _maybe_denormalize_xywh(bbox_xywh, (src_w, src_h))
        bbox_xyxy_224 = _resize_bbox(bbox_xywh, (src_w, src_h), (self.img_size, self.img_size))
        gt_map_224 = _bbox_to_gt_map(bbox_xyxy_224, size=self.img_size)

        gt: Dict[str, Any] = {
            'bbox_xyxy_224': torch.tensor(bbox_xyxy_224, dtype=torch.long),
            'gt_map_224': torch.from_numpy(gt_map_224).float(),
            'orig_size': (orig_h, orig_w),
            'meta': entry,
        }

        # 负样本选择策略：若存在类别字段，则从所有不同类别中随机挑选；否则随机不同索引
        if self.category_key is not None:
            cur_cat = entry[self.category_key]
            candidate_indices: List[int] = []
            for cat, ids in self.cat_to_indices.items():
                if cat != cur_cat:
                    candidate_indices.extend(ids)
            if len(candidate_indices) == 0:
                neg_idx = (idx + 1) % len(self.entries)
            else:
                neg_idx = int(np.random.choice(candidate_indices))
        else:
            # 无类别字段，随机挑一个不同索引
            neg_idx = (idx + np.random.randint(1, len(self.entries))) % len(self.entries)

        neg_entry = self.entries[neg_idx]
        neg_img_path = self._resolve_img_path(neg_entry)
        neg_coch_path = self._resolve_coch_path(neg_entry)
        neg_image_t, _ = self._load_image(neg_img_path)
        neg_coch_t = self._load_coch_npy(neg_coch_path)

        return image_t, audio_coch_t, gt, neg_image_t, neg_coch_t


def create_dataloader(
    config_json_path: str,
    image_root: str,
    audio_root: str,
    coch_config: Optional[Dict[str, Any]] = None,
    img_size: int = 224,
    source_wh: Tuple[int, int] = (1920, 1080),
    coch_target_ft: Optional[Tuple[int, int]] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = False,
    train: bool = False,
) -> Tuple[DataLoader, AudioCocoDataset]:
    """创建 DataLoader 及其底层 Dataset。

    coch_config 例如可来自 `cochleargram_config.py` 的 DEFAULT_CONFIG。
    """
    coch_config = coch_config or {}
    processor = AudioCocoProcessor(**coch_config)
    dataset = AudioCocoDataset(
        config_json_path=config_json_path,
        image_root=image_root,
        audio_root=audio_root,
        processor=processor,
        img_size=img_size,
        source_wh=source_wh,
        coch_target_ft=coch_target_ft,
        train=train,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader, dataset


def create_npy_dataloader(
    config_json_path: str,
    image_root: str,
    coch_root: str,
    img_size: int = 224,
    source_wh: Tuple[int, int] = (1920, 1080),
    coch_target_ft: Optional[Tuple[int, int]] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = False,
    train: bool = False,
) -> Tuple[DataLoader, AudioCocoNPYDataset]:
    """创建基于预生成 .npy cochleagram 的 DataLoader。"""
    dataset = AudioCocoNPYDataset(
        config_json_path=config_json_path,
        image_root=image_root,
        coch_root=coch_root,
        img_size=img_size,
        source_wh=source_wh,
        coch_target_ft=coch_target_ft,
        train=train,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader, dataset


__all__ = [
    'AudioCocoProcessor',
    'AudioCocoDataset',
    'AudioCocoNPYDataset',
    'create_dataloader',
    'create_npy_dataloader',
]


