import json
import os
import numpy as np
from copy import deepcopy
import io
from tqdm import tqdm
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

with open('/home/yanhao/SSHS/AudioCOCO/finalConfig/train.json', 'r') as file:
    train_data = json.load(file)

with open('/home/yanhao/SSHS/AudioCOCO/finalConfig/config1.json', 'r') as file:
    config1_data = json.load(file)

with open('/home/yanhao/SSHS/AudioCOCO/finalConfig/config2.json', 'r') as file:
    config2_data = json.load(file)

with open('/home/yanhao/SSHS/AudioCOCO/finalConfig/config3.json', 'r') as file:
    config3_data = json.load(file)

DEVICE = 'cuda:2'

depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitb14').to(DEVICE).eval()

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

def get_depth_value_for_point(image_path: str, point_xy: list) -> float:
    raw_image = cv2.imread(image_path)
    # 与旧流程保持一致，统一到 1920x1080 的坐标系
    raw_image = cv2.resize(raw_image, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    h, w = image.shape[:2]
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        depth = depth_anything(image)
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    # 最近像素（四舍五入到最近整数像素）
    x = int(round(float(point_xy[0])))
    y = int(round(float(point_xy[1])))
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    return float(depth[y, x].detach().cpu().item())

updated_data = []

for data in tqdm(train_data):
    MONITOR_DEPTH = 0.76

    image_name = data['image_id']
    image_path = '/home/yanhao/coco/train2014/' + image_name
    
    gt_box = data.get('gt_box', None)

    if gt_box is None:
        updated_data.append(data)
        continue

    point = [gt_box[0] + gt_box[2]/2, gt_box[1] + gt_box[3]/2]
    pre_depth_value = get_depth_value_for_point(image_path, point)
    new_depth_value = 1 - pre_depth_value

    unity_z = new_depth_value * 0.76 + MONITOR_DEPTH

    coords = list(data.get('coords', []))
    if len(coords) < 3:
        # 若 coords 不足 3 个元素，则填充到 3 个元素
        while len(coords) < 3:
            coords.append(0.0)
    coords[2] = unity_z
    new_data = deepcopy(data)
    new_data['coords'] = coords
    updated_data.append(new_data)

# 写入新的配置
output_config_path = '/home/yanhao/SSHS/AudioCOCO/finalConfig/train_depth.json'
with open(output_config_path, 'w') as f:
    json.dump(updated_data, f)

updated_data = []

for data in tqdm(config1_data):
    MONITOR_DEPTH = 0.76

    image_name = data['image_id']
    image_path = '/home/yanhao/coco/val2014/' + image_name
    
    gt_box = data.get('gt_box', None)

    if gt_box is None:
        updated_data.append(data)
        continue

    point = [gt_box[0] + gt_box[2]/2, gt_box[1] + gt_box[3]/2]
    pre_depth_value = get_depth_value_for_point(image_path, point)
    new_depth_value = 1 - pre_depth_value

    unity_z = new_depth_value * 0.76 + MONITOR_DEPTH

    coords = list(data.get('coords', []))
    if len(coords) < 3:
        # 若 coords 不足 3 个元素，则填充到 3 个元素
        while len(coords) < 3:
            coords.append(0.0)
    coords[2] = unity_z
    new_data = deepcopy(data)
    new_data['coords'] = coords
    updated_data.append(new_data)

# 写入新的配置
output_config_path = '/home/yanhao/SSHS/AudioCOCO/finalConfig/config1_depth.json'
with open(output_config_path, 'w') as f:
    json.dump(updated_data, f)

updated_data = []

for data in tqdm(config2_data):
    MONITOR_DEPTH = 0.76

    image_name = data['image_id']
    image_path = '/home/yanhao/coco/val2014/' + image_name
    
    gt_box = data.get('dist_gt_box', None)

    if gt_box is None:
        updated_data.append(data)
        continue

    point = [gt_box[0] + gt_box[2]/2, gt_box[1] + gt_box[3]/2]
    pre_depth_value = get_depth_value_for_point(image_path, point)
    new_depth_value = 1 - pre_depth_value

    unity_z = new_depth_value * 0.76 + MONITOR_DEPTH

    coords = list(data.get('coords', []))
    if len(coords) < 3:
        # 若 coords 不足 3 个元素，则填充到 3 个元素
        while len(coords) < 3:
            coords.append(0.0)
    coords[2] = unity_z
    new_data = deepcopy(data)
    new_data['coords'] = coords
    updated_data.append(new_data)

# 写入新的配置
output_config_path = '/home/yanhao/SSHS/AudioCOCO/finalConfig/config2_depth.json'
with open(output_config_path, 'w') as f:
    json.dump(updated_data, f)

updated_data = []

for data in tqdm(config3_data):
    MONITOR_DEPTH = 0.76

    image_name = data['image_id']
    image_path = '/home/yanhao/coco/val2014/' + image_name
    
    gt_box = data.get('gt_box', None)

    if gt_box is None:
        updated_data.append(data)
        continue

    point = [gt_box[0] + gt_box[2]/2, gt_box[1] + gt_box[3]/2]
    pre_depth_value = get_depth_value_for_point(image_path, point)
    new_depth_value = 1 - pre_depth_value

    unity_z = new_depth_value * 0.76 + MONITOR_DEPTH

    coords = list(data.get('coords', []))
    if len(coords) < 3:
        # 若 coords 不足 3 个元素，则填充到 3 个元素
        while len(coords) < 3:
            coords.append(0.0)
    coords[2] = unity_z
    new_data = deepcopy(data)
    new_data['coords'] = coords
    updated_data.append(new_data)

# 写入新的配置
output_config_path = '/home/yanhao/SSHS/AudioCOCO/finalConfig/config3_depth.json'
with open(output_config_path, 'w') as f:
    json.dump(updated_data, f)