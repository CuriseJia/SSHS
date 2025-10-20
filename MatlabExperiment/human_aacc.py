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


def get_mask_from_annotation(annotation: Dict, scale_x: float, scale_y: float) -> np.ndarray:
    """从标注中获取缩放后的mask"""
    if 'segmentation' in annotation:
        segmentation = annotation['segmentation']
        
        # 处理RLE格式的segmentation
        if isinstance(segmentation, dict) and 'counts' in segmentation and 'size' in segmentation:
            # 缩放mask尺寸到1920x1080
            original_mask_size = segmentation['size']
            scaled_mask_size = (1080, 1920)  # (height, width)
            
            # 缩放RLE counts
            scaled_counts = scale_rle_counts(segmentation['counts'], original_mask_size, scaled_mask_size)
            return decode_rle_mask(scaled_counts, scaled_mask_size)
        # 处理polygon格式的segmentation
        elif isinstance(segmentation, list) and len(segmentation) > 0:
            # 缩放polygon坐标
            scaled_segmentation = scale_polygon(segmentation, scale_x, scale_y)
            if scaled_segmentation:
                return polygon_to_mask(scaled_segmentation, 1080, 1920)
    
    return None


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


def calculate_bbox_center(bx: float, by: float, bw: float, bh: float) -> Tuple[float, float]:
    """计算bbox的中心点"""
    center_x = bx + bw / 2
    center_y = by + bh / 2
    return center_x, center_y


def calculate_mask_center(mask: np.ndarray) -> Tuple[float, float]:
    """计算mask的中心点"""
    if mask is None:
        return 0, 0
    
    # 找到mask中所有非零像素的坐标
    y_coords, x_coords = np.where(mask > 0)
    
    if len(x_coords) == 0:
        return 0, 0
    
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    return center_x, center_y


def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算两点之间的距离"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def find_closest_mask_to_bbox(annotations: List[Dict], bbox_center: Tuple[float, float], 
                             scale_x: float, scale_y: float) -> np.ndarray:
    """找到距离bbox中心点最近的mask"""
    closest_mask = None
    min_distance = float('inf')
    
    for annotation in annotations:
        mask = get_mask_from_annotation(annotation, scale_x, scale_y)
        if mask is not None:
            mask_center = calculate_mask_center(mask)
            distance = calculate_distance(bbox_center, mask_center)
            
            if distance < min_distance:
                min_distance = distance
                closest_mask = mask
    
    return closest_mask


def point_in_bbox(x: float, y: float, bx: float, by: float, bw: float, bh: float) -> bool:
	"""Return True if (x, y) is inside the bbox defined by top-left (bx, by) and size (bw, bh)."""
	return (bx <= x <= bx + bw) and (by <= y <= by + bh)


def compute_accuracy_from_files(
	directory_path: str,
	coco_json_path: str,
	tasks_to_report = (1, 2, 3, 4, 6),
	object_sizes = ("size1", "size2", "size3"),
) -> Tuple[Dict[Tuple[int, str], float], Dict[Tuple[int, str], int], Dict[Tuple[int, str], int]]:
	"""从指定目录下的所有.txt文件读取数据并计算准确率
	
	Returns:
		tuple: (accuracy_dict, correct_counts, total_counts)
	"""
	# 加载COCO数据
	coco_data = load_coco_annotations(coco_json_path)
	image_annotations, category_id_to_name = create_image_annotation_mapping(coco_data)
	
	print(f"加载了 {len(image_annotations)} 个图像的标注信息")
	
	correct_counts: Dict[Tuple[int, str], int] = defaultdict(int)
	total_counts: Dict[Tuple[int, str], int] = defaultdict(int)
	
	# 获取目录下所有.txt文件
	txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
	print(f"找到 {len(txt_files)} 个.txt文件: {[os.path.basename(f) for f in txt_files]}")
	
	# 遍历所有.txt文件
	for file_path in tqdm(txt_files, desc="处理文件"):
		print(f"正在处理文件: {os.path.basename(file_path)}")
		try:
			with open(file_path, "r", newline="", encoding="utf-8") as f:
				reader = csv.reader(f, delimiter="\t")
				head = next(reader, None)  # header
				file_correct = 0
				file_total = 0
				
				for row in reader:
					# Expected columns per header:
					# x, y, image, audio, category, object_size, gt_box (4 values), task, time
					if not row or len(row) < 11:
						continue

					try:
						x = float(row[0])
						y = float(row[1])
						image_name = row[2]
						category = row[4]
						obj_size = row[5]
						bx = float(row[6])
						by = float(row[7])
						bw = float(row[8])
						bh = float(row[9])
						task = int(row[10])
					except (ValueError, IndexError):
						# Skip malformed data
						continue

					if task == 4:
						if "gaussian_noise_image" in image_name:
							# 高斯噪声图像组
							noise_key = (task, f"{obj_size}_gaussian_noise")
							total_counts[noise_key] += 1
							file_total += 1
							if point_in_bbox(x, y, bx, by, bw, bh):
								correct_counts[noise_key] += 1
								file_correct += 1
						elif "black_image.jpg" in image_name:
							# 黑色图像组
							black_key = (task, f"{obj_size}_black_image")
							total_counts[black_key] += 1
							file_total += 1
							if point_in_bbox(x, y, bx, by, bw, bh):
								correct_counts[black_key] += 1
								file_correct += 1
						else:
							# 其他图像类型，使用原始key
							key = (task, obj_size)
							total_counts[key] += 1
							file_total += 1
							if point_in_bbox(x, y, bx, by, bw, bh):
								correct_counts[key] += 1
								file_correct += 1
					else:
						# 其他task使用原始key
						key = (task, obj_size)
						total_counts[key] += 1
						file_total += 1
					
					if task != 4 and image_name in image_annotations:
						annotations = image_annotations[image_name]
						
						coco_category = "airplane" if category == "plane" else category
						
						# 查找匹配类别的标注
						matching_annotations = []
						for ann in annotations:
							if ann['category_name'] == coco_category:
								matching_annotations.append(ann)
						
						if matching_annotations:
							# 获取原图尺寸
							original_height = matching_annotations[0]['image_info']['height']
							original_width = matching_annotations[0]['image_info']['width']
							
							# 计算缩放比例 (原图 -> 1920x1080)
							scale_x = 1920.0 / original_width
							scale_y = 1080.0 / original_height
							
							if task == 6:
								# Task 6: 选择距离gt_box中心点最近的mask
								bbox_center = calculate_bbox_center(bx, by, bw, bh)
								scaled_bbox_center = (bbox_center[0] * scale_x, bbox_center[1] * scale_y)
								
								# 找到距离bbox中心点最近的mask
								closest_mask = find_closest_mask_to_bbox(matching_annotations, scaled_bbox_center, scale_x, scale_y)
								
								if closest_mask is not None and point_in_mask(x, y, closest_mask):
									correct_counts[key] += 1
									file_correct += 1
							else:
								# Task 1,2,3: 合并所有同类mask
								combined_mask = None
								
								for matching_annotation in matching_annotations:
									current_mask = get_mask_from_annotation(matching_annotation, scale_x, scale_y)
									if current_mask is not None:
										if combined_mask is None:
											combined_mask = current_mask.copy()
										else:
											# 合并mask（使用OR操作）
											combined_mask = np.logical_or(combined_mask, current_mask).astype(np.uint8)
								
								if combined_mask is not None and point_in_mask(x, y, combined_mask):
									correct_counts[key] += 1
									file_correct += 1
				
				print(f"  - 文件 {os.path.basename(file_path)}: {file_correct}/{file_total} 正确")
				
		except Exception as e:
			print(f"处理文件 {file_path} 时出错: {e}")
			continue

	# Compute accuracy
	accuracy: Dict[Tuple[int, str], float] = {}
	for task in tasks_to_report:
		if task == 4:
			# Task 4 需要计算特殊的分组key
			for size in object_sizes:
				# 高斯噪声图像组
				noise_key = (task, f"{size}_gaussian_noise")
				c = correct_counts.get(noise_key, 0)
				t = total_counts.get(noise_key, 0)
				acc = (c / t) if t > 0 else 0.0
				accuracy[noise_key] = acc
				
				# 黑色图像组
				black_key = (task, f"{size}_black_image")
				c = correct_counts.get(black_key, 0)
				t = total_counts.get(black_key, 0)
				acc = (c / t) if t > 0 else 0.0
				accuracy[black_key] = acc
		else:
			# 其他task使用原始key
			for size in object_sizes:
				key = (task, size)
				c = correct_counts.get(key, 0)
				t = total_counts.get(key, 0)
				acc = (c / t) if t > 0 else 0.0
				accuracy[key] = acc

	return accuracy, correct_counts, total_counts


def main() -> None:
	directory_path = "/home/yanhao/SSHS/MatlabExperiment"
	coco_json_path = "/home/yanhao/SSHS/AudioCOCO/instances_val2014.json"
	accuracy, correct_counts, total_counts = compute_accuracy_from_files(directory_path, coco_json_path)

	# Print as a compact table: tasks 1,2,3,4,6 × size1,size2,size3
	print("\n=== 合并所有.txt文件后的准确率结果 ===")
	print("task\tsize1\tsize2\tsize3")
	for task in (1, 2, 3, 4, 6):
		vals = [
			f"{accuracy.get((task, 'size1'), 0.0):.3f}",
			f"{accuracy.get((task, 'size2'), 0.0):.3f}",
			f"{accuracy.get((task, 'size3'), 0.0):.3f}",
		]
		print(f"{task}\t" + "\t".join(vals))
	
	# 打印详细统计信息
	print("\n=== 详细统计信息 ===")
	for task in (1, 2, 3, 4, 6):
		print(f"\nTask {task}:")
		if task == 4:
			# Task 4 按图像类型分组显示
			print("  高斯噪声图像组:")
			for size in ("size1", "size2", "size3"):
				key = (task, f"{size}_gaussian_noise")
				correct = correct_counts.get(key, 0)
				total = total_counts.get(key, 0)
				acc = accuracy.get(key, 0.0)
				print(f"    {size}: {correct}/{total} = {acc:.3f} ({acc*100:.1f}%)")
			
			print("  黑色图像组:")
			for size in ("size1", "size2", "size3"):
				key = (task, f"{size}_black_image")
				correct = correct_counts.get(key, 0)
				total = total_counts.get(key, 0)
				acc = accuracy.get(key, 0.0)
				print(f"    {size}: {correct}/{total} = {acc:.3f} ({acc*100:.1f}%)")
			
		else:
			# 其他task正常显示
			for size in ("size1", "size2", "size3"):
				key = (task, size)
				correct = correct_counts.get(key, 0)
				total = total_counts.get(key, 0)
				acc = accuracy.get(key, 0.0)
				print(f"  {size}: {correct}/{total} = {acc:.3f} ({acc*100:.1f}%)")


if __name__ == "__main__":
	main()


