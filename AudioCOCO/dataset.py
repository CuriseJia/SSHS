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


class AudioCocoProcessor:
    """将单通道音频波形或wav路径转换为 cochleagram 的处理器。

    - 若输入为路径: 自动加载并转单声道
    - 若输入为数组: 视为单通道波形, 需提供采样率
    - 返回: np.ndarray, 形状约为 (n_filters * sample_factor, time)
    """

    def __init__(self, **coch_config: Any) -> None:
        self.preprocessor = CochleagramPreprocessor(**coch_config)

    def __call__(self, audio: Union[str, np.ndarray], sr: Optional[int] = None) -> np.ndarray:
        if isinstance(audio, str):
            signal, file_sr = self.preprocessor.load_audio(audio)
            coch = self.preprocessor.generate_cochleagram(signal, file_sr)
            return coch
        else:
            assert sr is not None, "当传入numpy波形时必须提供采样率sr"
            # 若为多通道, 取平均到单通道
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            coch = self.preprocessor.generate_cochleagram(audio, sr)
            return coch


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


class AudioCocoDataset(Dataset):
    """按照 config1.json 的条目, 返回图像-音频对及其GT。

    每个样本返回:
    - image: FloatTensor [3, H, W] (默认224)
    - audio_coch: FloatTensor [1, F, T]
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
        train: bool = False,
    ) -> None:
        super().__init__()
        self.image_root = image_root
        self.audio_root = audio_root
        self.processor = processor
        self.img_size = img_size
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
        coch = self.processor(path)  # np.ndarray [F, T]
        # 转为 torch [1, F, T]
        coch_t = torch.from_numpy(coch).float().unsqueeze(0)
        return coch_t

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        entry = self.entries[idx]
        img_path, aud_path = self._resolve_paths(entry)

        image_t, (orig_h, orig_w) = self._load_image(img_path)
        audio_coch_t = self._load_audio_coch(aud_path)

        bbox_xywh: List[float] = entry['gt_box']
        bbox_xyxy_224 = _resize_bbox(bbox_xywh, (orig_w, orig_h), (self.img_size, self.img_size))
        gt_map_224 = _bbox_to_gt_map(bbox_xyxy_224, size=self.img_size)

        gt: Dict[str, Any] = {
            'bbox_xyxy_224': torch.tensor(bbox_xyxy_224, dtype=torch.long),
            'gt_map_224': torch.from_numpy(gt_map_224).float(),
            'orig_size': (orig_h, orig_w),
            'meta': entry,
        }

        return image_t, audio_coch_t, gt


def create_dataloader(
    config_json_path: str,
    image_root: str,
    audio_root: str,
    coch_config: Optional[Dict[str, Any]] = None,
    img_size: int = 224,
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
        train=train,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader, dataset


__all__ = [
    'AudioCocoProcessor',
    'AudioCocoDataset',
    'create_dataloader',
]


