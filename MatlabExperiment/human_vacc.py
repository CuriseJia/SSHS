#!/usr/bin/env python3
"""
human_vacc.py - 计算Task 6的视觉准确率(VACC)

功能：
1. 读取路径下所有的.txt文件中task列为6的数据
2. 根据task6数据的image和category信息读取instances_val2014.json中对应图像的category mask
3. 若task6数据的x和y位于mask中则认为正确
4. 分别按object_size报告正确率
"""

import csv
import os
import glob
import json
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List
from tqdm import tqdm

# 尝试导入pycocotools，如果不可用则使用备用方法
try:
    from pycocotools import mask as mask_utils
    PYOCO_AVAILABLE = True
except ImportError:
    print("警告: pycocotools不可用，将使用备用的RLE解码方法")
    PYOCO_AVAILABLE = False

# 尝试导入matplotlib，如果不可用则使用备用方法
try:
    from matplotlib.path import Path
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("警告: matplotlib不可用，将使用简化的polygon处理方法")
    MATPLOTLIB_AVAILABLE = False

# 尝试导入cv2，如果不可用则使用备用方法
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("警告: cv2不可用，将使用简化的图像缩放方法")
    CV2_AVAILABLE = False


def load_coco_annotations(json_path: str) -> Dict:
    """加载COCO标注文件"""
    print(f"正在加载COCO标注文件: {json_path}")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data


def create_image_annotation_mapping(coco_data: Dict) -> Tuple[Dict[str, List[Dict]], Dict[str, int]]:
    """创建图像文件名到标注列表的映射"""
    image_id_to_annotations = defaultdict(list)
    
    # 创建图像ID到图像信息的映射
    image_id_to_info = {}
    for img in coco_data['images']:
        image_id_to_info[img['id']] = img
    
    # 创建类别ID到名称的映射
    category_id_to_name = {}
    for cat in coco_data['categories']:
        category_id_to_name[cat['id']] = cat['name']
    
    # 为每个标注创建映射
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        image_info = image_id_to_info.get(image_id)
        if image_info:
            # 添加图像信息和类别名称到标注中
            ann_with_image = ann.copy()
            ann_with_image['image_info'] = image_info
            ann_with_image['category_name'] = category_id_to_name.get(ann['category_id'])
            image_id_to_annotations[image_info['file_name']].append(ann_with_image)
    
    return image_id_to_annotations, category_id_to_name


def decode_rle_mask(rle_counts: List[int], size: Tuple[int, int]) -> np.ndarray:
    """解码RLE格式的mask"""
    try:
        if PYOCO_AVAILABLE:
            # 使用pycocotools解码RLE
            rle = {'counts': rle_counts, 'size': size}
            mask = mask_utils.decode(rle)
            return mask
        else:
            # 使用备用方法解码RLE
            return decode_rle_manual(rle_counts, size)
    except Exception as e:
        print(f"解码RLE mask时出错: {e}")
        return None


def decode_rle_manual(rle_counts: List[int], size: Tuple[int, int]) -> np.ndarray:
    """手动解码RLE格式的mask（备用方法）"""
    try:
        height, width = size
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 将RLE counts转换为像素值
        pixel_values = []
        for i, count in enumerate(rle_counts):
            value = 1 if i % 2 == 0 else 0  # 奇数索引为0，偶数索引为1
            pixel_values.extend([value] * count)
        
        # 将像素值填充到mask中
        if len(pixel_values) >= height * width:
            for i in range(height):
                for j in range(width):
                    idx = i * width + j
                    if idx < len(pixel_values):
                        mask[i, j] = pixel_values[idx]
        
        return mask
    except Exception as e:
        print(f"手动解码RLE时出错: {e}")
        return None


def _fill_polygon_in_mask(mask: np.ndarray, points: List[List[float]], height: int, width: int):
    """在多边形内填充mask"""
    try:
        if len(points) >= 3:
            if MATPLOTLIB_AVAILABLE:
                # 使用matplotlib进行精确的多边形填充
                path = Path(points)
                
                # 创建网格
                y, x = np.mgrid[:height, :width]
                points_grid = np.column_stack((x.ravel(), y.ravel()))
                
                # 检查哪些点在多边形内
                inside = path.contains_points(points_grid)
                inside = inside.reshape((height, width))
                
                mask[inside] = 1
            else:
                # 使用简化的方法：只检查边界框内的点
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                min_x, max_x = int(min(x_coords)), int(max(x_coords))
                min_y, max_y = int(min(y_coords)), int(max(y_coords))
                
                # 确保在图像范围内
                min_x = max(0, min_x)
                max_x = min(width, max_x)
                min_y = max(0, min_y)
                max_y = min(height, max_y)
                
                # 简单的点在多边形内检查（射线法）
                for y in range(min_y, max_y):
                    for x in range(min_x, max_x):
                        if point_in_polygon(x, y, points):
                            mask[y, x] = 1
    except Exception as e:
        print(f"填充多边形时出错: {e}")


def polygon_to_mask(polygons: List, height: int, width: int) -> np.ndarray:
    """将polygon格式转换为mask"""
    try:
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 处理嵌套列表格式的polygon数据
        if len(polygons) > 0 and isinstance(polygons[0], list) and len(polygons[0]) > 0 and isinstance(polygons[0][0], list):
            # 如果polygons是嵌套列表格式 [[[x1,y1,x2,y2,...], [x1,y1,x2,y2,...]], ...]
            for polygon_group in polygons:
                for polygon in polygon_group:
                    if len(polygon) >= 6:  # 至少需要3个点
                        points = []
                        for i in range(0, len(polygon), 2):
                            if i + 1 < len(polygon):
                                try:
                                    x_val = float(polygon[i])
                                    y_val = float(polygon[i + 1])
                                    points.append([x_val, y_val])
                                except (ValueError, TypeError):
                                    continue
                        if len(points) >= 3:
                            _fill_polygon_in_mask(mask, points, height, width)
        else:
            # 处理平铺列表格式的polygon数据
            for polygon in polygons:
                if len(polygon) >= 6:  # 至少需要3个点
                    # 将polygon转换为点对
                    points = []
                    for i in range(0, len(polygon), 2):
                        if i + 1 < len(polygon):
                            try:
                                x_val = float(polygon[i])
                                y_val = float(polygon[i + 1])
                                points.append([x_val, y_val])
                            except (ValueError, TypeError):
                                continue
                    if len(points) >= 3:
                        _fill_polygon_in_mask(mask, points, height, width)
        
        return mask
    except Exception as e:
        print(f"转换polygon到mask时出错: {e}")
        return None


def point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    """使用射线法检查点是否在多边形内"""
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def scale_polygon(polygon: List, scale_x: float, scale_y: float) -> List:
    """缩放polygon坐标"""
    try:
        scaled_polygon = []
        
        # 处理嵌套列表格式的polygon数据
        if len(polygon) > 0 and isinstance(polygon[0], list):
            # 如果polygon是嵌套列表格式 [[x1,y1,x2,y2,...], [x1,y1,x2,y2,...], ...]
            for poly in polygon:
                if isinstance(poly, list) and len(poly) >= 6:
                    scaled_poly = []
                    for i in range(0, len(poly), 2):
                        if i + 1 < len(poly):
                            try:
                                x_val = float(poly[i])
                                y_val = float(poly[i + 1])
                                scaled_x = x_val * scale_x
                                scaled_y = y_val * scale_y
                                scaled_poly.extend([scaled_x, scaled_y])
                            except (ValueError, TypeError):
                                continue
                    if scaled_poly:
                        scaled_polygon.append(scaled_poly)
        else:
            # 处理平铺列表格式的polygon数据 [x1,y1,x2,y2,...]
            for i in range(0, len(polygon), 2):
                if i + 1 < len(polygon):
                    try:
                        x_val = float(polygon[i])
                        y_val = float(polygon[i + 1])
                        scaled_x = x_val * scale_x
                        scaled_y = y_val * scale_y
                        scaled_polygon.extend([scaled_x, scaled_y])
                    except (ValueError, TypeError):
                        continue
        
        return scaled_polygon
    except Exception as e:
        print(f"缩放polygon时出错: {e}")
        return polygon


def scale_rle_counts(counts: List[int], original_size: Tuple[int, int], target_size: Tuple[int, int]) -> List[int]:
    """缩放RLE counts以适应新的尺寸"""
    try:
        if PYOCO_AVAILABLE and CV2_AVAILABLE:
            # 使用pycocotools和cv2进行精确缩放
            original_height, original_width = original_size
            target_height, target_width = target_size
            
            # 解码原始mask
            original_rle = {'counts': counts, 'size': original_size}
            original_mask = mask_utils.decode(original_rle)
            
            # 缩放mask
            scaled_mask = cv2.resize(original_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            
            # 重新编码为RLE
            scaled_rle = mask_utils.encode(np.asfortranarray(scaled_mask))
            return scaled_rle['counts']
        else:
            # 简化方法：直接返回原始counts（可能不够精确）
            print("警告: 无法精确缩放RLE，使用原始数据")
            return counts
    except Exception as e:
        print(f"缩放RLE counts时出错: {e}")
        return counts


def point_in_mask(x: float, y: float, mask: np.ndarray) -> bool:
    """检查点(x,y)是否在mask中"""
    if mask is None:
        return False
    
    # 确保坐标在图像范围内
    h, w = mask.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    
    # 检查像素值
    return mask[int(y), int(x)] > 0


def compute_vacc_accuracy(
    directory_path: str,
    coco_json_path: str,
    object_sizes: Tuple[str, ...] = ("size1", "size2", "size3")
) -> Dict[str, float]:
    """计算Task 6的视觉准确率"""
    
    # 加载COCO数据
    coco_data = load_coco_annotations(coco_json_path)
    image_annotations, category_id_to_name = create_image_annotation_mapping(coco_data)
    
    print(f"加载了 {len(image_annotations)} 个图像的标注信息")
    
    # 统计变量 - 只按对象大小统计
    correct_counts: Dict[str, int] = defaultdict(int)
    total_counts: Dict[str, int] = defaultdict(int)
    
    # 获取所有.txt文件
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
    print(f"找到 {len(txt_files)} 个.txt文件")
    
    # 处理所有文件
    total_processed = 0
    total_correct = 0
    
    for file_path in tqdm(txt_files, desc="处理文件"):
        try:
            with open(file_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                header = next(reader, None)  # 跳过表头
                
                for row in reader:
                    if not row or len(row) < 11:
                        continue
                    
                    try:
                        x = float(row[0])
                        y = float(row[1])
                        image_name = row[2]
                        category = row[4]
                        obj_size = row[5]
                        task = int(row[10])
                        
                        # 只处理task 6的数据
                        if task != 6:
                            continue
                            
                    except (ValueError, IndexError):
                        continue
                    
                    # 检查图像是否在COCO数据中
                    if image_name not in image_annotations:
                        continue
                    
                    # 获取该图像的所有标注
                    annotations = image_annotations[image_name]
                    
                    # 查找所有匹配类别的标注
                    matching_annotations = []
                    for ann in annotations:
                        if ann['category_name'] == category:
                            matching_annotations.append(ann)
                    
                    if not matching_annotations:
                        continue
                    
                    # 调试信息：显示找到的匹配标注数量
                    # if len(matching_annotations) > 1:
                    #     print(f"图像 {image_name} 中找到 {len(matching_annotations)} 个 {category} 类别的标注，将合并处理")
                    
                    # 获取原图尺寸（从第一个标注获取，所有标注应该来自同一图像）
                    original_height = matching_annotations[0]['image_info']['height']
                    original_width = matching_annotations[0]['image_info']['width']
                    
                    # 计算缩放比例 (原图 -> 1920x1080)
                    scale_x = 1920.0 / original_width
                    scale_y = 1080.0 / original_height
                    
                    # 合并所有匹配类别的mask
                    combined_mask = None
                    
                    for matching_annotation in matching_annotations:
                        if 'segmentation' in matching_annotation:
                            segmentation = matching_annotation['segmentation']
                            
                            # 处理RLE格式的segmentation
                            if isinstance(segmentation, dict) and 'counts' in segmentation and 'size' in segmentation:
                                # 缩放mask尺寸到1920x1080
                                original_mask_size = segmentation['size']
                                scaled_mask_size = (1080, 1920)  # (height, width)
                                
                                # 缩放RLE counts
                                scaled_counts = scale_rle_counts(segmentation['counts'], original_mask_size, scaled_mask_size)
                                current_mask = decode_rle_mask(scaled_counts, scaled_mask_size)
                            # 处理polygon格式的segmentation
                            elif isinstance(segmentation, list) and len(segmentation) > 0:
                                # 缩放polygon坐标
                                scaled_segmentation = scale_polygon(segmentation, scale_x, scale_y)
                                if scaled_segmentation:
                                    current_mask = polygon_to_mask(scaled_segmentation, 1080, 1920)
                                else:
                                    continue
                            else:
                                continue
                            
                            if current_mask is not None:
                                if combined_mask is None:
                                    combined_mask = current_mask.copy()
                                else:
                                    # 合并mask（使用OR操作）
                                    combined_mask = np.logical_or(combined_mask, current_mask).astype(np.uint8)
                    
                    if combined_mask is not None:
                        # 检查缩放后的点是否在合并后的mask中
                        total_counts[obj_size] += 1
                        total_processed += 1
                        
                        if point_in_mask(x, y, combined_mask):
                            correct_counts[obj_size] += 1
                            total_correct += 1
                        
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue
    
    print(f"总共处理了 {total_processed} 个Task 6数据点")
    
    # 计算准确率 - 只按对象大小
    accuracy: Dict[str, float] = {}
    for size in object_sizes:
        correct = correct_counts.get(size, 0)
        total = total_counts.get(size, 0)
        acc = (correct / total) if total > 0 else 0.0
        accuracy[size] = acc
    
    return accuracy, correct_counts, total_counts


def main():
    """主函数"""
    # 设置路径
    directory_path = "/home/yanhao/SSHS/MatlabExperiment"
    coco_json_path = "/home/yanhao/SSHS/AudioCOCO/instances_val2014.json"
    
    print("=== Task 6 视觉准确率(VACC)计算 ===")
    
    # 计算准确率
    accuracy, correct_counts, total_counts = compute_vacc_accuracy(
        directory_path, coco_json_path
    )
    
    # 打印结果
    print("\n=== Task 6 视觉准确率结果 ===")
    print("object_size\t正确数\t总样本数\t准确率")
    
    sizes = ["size1", "size2", "size3"]
    total_correct_all = 0
    total_samples_all = 0
    
    for size in sizes:
        correct = correct_counts.get(size, 0)
        total = total_counts.get(size, 0)
        acc = accuracy.get(size, 0.0)
        print(f"{size}\t\t{correct}\t{total}\t\t{acc:.3f} ({acc*100:.1f}%)")
        total_correct_all += correct
        total_samples_all += total
    
    # 总体准确率
    overall_accuracy = (total_correct_all / total_samples_all) if total_samples_all > 0 else 0.0
    print(f"总体\t\t{total_correct_all}\t{total_samples_all}\t\t{overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")


if __name__ == "__main__":
    main()
