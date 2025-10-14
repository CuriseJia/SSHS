import json
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import os
from collections import defaultdict

def format_image_filename(annotation_file_path: str, img_id: int) -> str:
    """Format image filename based on whether annotation file is train or val."""
    prefix = "COCO_val2014_"
    lower_path = (annotation_file_path or "").lower()
    if "train2014" in lower_path:
        prefix = "COCO_train2014_"
    return f"{prefix}{img_id:012d}.jpg"

def load_categories(category_file):
    """Load category file"""
    with open(category_file, 'r') as f:
        categories = [line.strip() for line in f.readlines()]
    return categories

def calculate_mask_area_ratio(mask, image_width, image_height):
    """Calculate the ratio of mask area to total image area"""
    total_pixels = image_width * image_height
    mask_pixels = np.sum(mask)
    return mask_pixels / total_pixels

def get_object_size(ratio):
    """Determine object_size based on mask area ratio"""
    if ratio <= 0.05:
        return "size1"
    elif ratio <= 0.15:
        return "size2"
    else:
        return "size3"

def resize_image_and_annotations(image, gt_box, mask, target_width=1920, target_height=1080):
    """Resize image to target size and adjust gt_box and mask accordingly"""
    h, w = image.shape[:2]
    
    # Calculate scaling ratio
    scale_x = target_width / w
    scale_y = target_height / h
    
    # Resize image
    resized_image = cv2.resize(image, (target_width, target_height))
    
    # Adjust gt_box coordinates
    resized_gt_box = [
        int(gt_box[0] * scale_x),  # x
        int(gt_box[1] * scale_y),  # y
        int(gt_box[2] * scale_x),  # width
        int(gt_box[3] * scale_y)   # height
    ]
    
    # Adjust mask
    resized_mask = cv2.resize(mask.astype(np.uint8), (target_width, target_height))
    resized_mask = resized_mask.astype(bool)
    
    return resized_image, resized_gt_box, resized_mask

def calculate_center_point(gt_box):
    """Calculate the center point of gt_box"""
    x, y, w, h = gt_box
    center_x = x + w / 2
    center_y = y + h / 2
    return [center_x, center_y]

def filter_coco_instances(coco_annotation_file, category_file, output_file):
    """
    Filter images based on COCO dataset format
    
    Parameters:
    coco_annotation_file: COCO annotation file path (instances_val2014.json)
    category_file: Category file path (category.txt)
    output_file: Output JSON file path
    
    Output format:
    {
        "image_id": "COCO_val2014_000000000073.jpg",
        "category": "dog",
        "object_size": "size2",
        "gt_box": [100, 200, 300, 400],
        "point": [250, 400]
    }
    """
    # Load COCO dataset
    coco = COCO(coco_annotation_file)
    
    # Load target categories
    target_categories = load_categories(category_file)
    print(f"Target categories: {target_categories}")
    
    # Get category ID mapping
    cat_ids = coco.getCatIds(catNms=target_categories)
    print(f"Found category IDs: {cat_ids}")
    
    # Get all image IDs
    img_ids = coco.getImgIds()
    print(f"Total image count: {len(img_ids)}")
    
    # Store instances that meet the criteria
    filtered_instances = []
    
    # Statistics
    stats = {
        'total_images': len(img_ids),
        'processed_images': 0,
        'valid_images': 0,
        'category_counts': defaultdict(int),
        'size_counts': defaultdict(int)
    }
    
    for img_id in img_ids:
        stats['processed_images'] += 1
        
        if stats['processed_images'] % 1000 == 0:
            print(f"Processed {stats['processed_images']}/{stats['total_images']} images")
        
        # Get image information
        img_info = coco.loadImgs(img_id)[0]
        image_width = img_info['width']
        image_height = img_info['height']
        
        # Get all annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        annotations = coco.loadAnns(ann_ids)
        
        if not annotations:
            continue
        
        # Group instances in this image by category
        category_counts = defaultdict(int)
        for ann in annotations:
            cat_id = ann['category_id']
            cat_info = coco.loadCats(cat_id)[0]
            category_name = cat_info['name']
            category_counts[category_name] += 1
        
        # Check if there is only one instance of target category
        valid_categories = [cat for cat in category_counts.keys() if cat in target_categories]
        if len(valid_categories) != 1:
            continue  # Skip images with multiple target categories or no target categories
        
        category_name = valid_categories[0]
        if category_counts[category_name] != 1:
            continue  # Skip images with multiple instances of this category
        
        # Find the annotation for this instance
        target_ann = None
        for ann in annotations:
            cat_id = ann['category_id']
            cat_info = coco.loadCats(cat_id)[0]
            if cat_info['name'] == category_name:
                target_ann = ann
                break
        
        if target_ann is None:
            continue
        
        # Decode mask
        mask = coco.annToMask(target_ann)
        
        # Calculate mask area ratio
        area_ratio = calculate_mask_area_ratio(mask, image_width, image_height)
        
        # Determine object_size
        object_size = get_object_size(area_ratio)
        
        # Get gt_box (COCO format: [x, y, width, height])
        gt_box = target_ann['bbox']
        
        # Simulate image resize (here we only adjust coordinates, not actually process images)
        # In actual applications, you may need to load actual images for resize
        resized_gt_box = [
            int(gt_box[0] * 1920 / image_width),   # x
            int(gt_box[1] * 1080 / image_height),  # y
            int(gt_box[2] * 1920 / image_width),   # width
            int(gt_box[3] * 1080 / image_height)   # height
        ]
        
        # Calculate center point
        center_point = calculate_center_point(resized_gt_box)
        
        # Create instance record
        # Convert numeric ID to COCO format filename (auto train/val)
        image_filename = format_image_filename(coco_annotation_file, img_id)
        
        instance_record = {
            "image_id": image_filename,
            "category": category_name,
            "object_size": object_size,
            "gt_box": resized_gt_box,
            "point": center_point
        }
        
        filtered_instances.append(instance_record)
        stats['valid_images'] += 1
        stats['category_counts'][category_name] += 1
        stats['size_counts'][object_size] += 1
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_instances, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print(f"\nFiltering completed!")
    print(f"Total images: {stats['total_images']}")
    print(f"Valid images: {stats['valid_images']}")
    print(f"Filtering ratio: {stats['valid_images']/stats['total_images']*100:.2f}%")
    
    print(f"\nCategory statistics:")
    for category, count in stats['category_counts'].items():
        print(f"  {category}: {count}")
    
    print(f"\nSize statistics:")
    for size, count in stats['size_counts'].items():
        print(f"  {size}: {count}")
    
    return filtered_instances

def filter_coco_multi_instances(coco_annotation_file, category_file, output_file_multi):
    """
    Filter images where there are multiple instances of the same target category in one image.
    - For instances of categories listed in category.txt with count >= 2 in an image
    - Compute object_size by mask area ratio (0-5%: size1, 5-15%: size2, 15-25%: size3; >25% also treated as size3)
    - Resize geometry to 1920x1080 (only scale boxes), compute center point
    - Keep only 2 instance records per image (largest mask ratio)
    """
    coco = COCO(coco_annotation_file)
    target_categories = load_categories(category_file)
    cat_ids = coco.getCatIds(catNms=target_categories)

    img_ids = coco.getImgIds()
    results = []

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        image_width = img_info['width']
        image_height = img_info['height']

        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue

        # Count per-category instances in this image among target categories
        per_cat_anns = defaultdict(list)
        for ann in anns:
            cat_name = coco.loadCats(ann['category_id'])[0]['name']
            if cat_name in target_categories:
                per_cat_anns[cat_name].append(ann)

        # Collect candidate instance records for this image
        image_candidates = []
        for cat_name, cat_anns in per_cat_anns.items():
            if len(cat_anns) < 2:
                continue
            for ann in cat_anns:
                mask = coco.annToMask(ann)
                area_ratio = calculate_mask_area_ratio(mask, image_width, image_height)
                object_size = get_object_size(area_ratio)
                x, y, w, h = ann['bbox']
                resized_gt_box = [
                    int(x * 1920 / image_width),
                    int(y * 1080 / image_height),
                    int(w * 1920 / image_width),
                    int(h * 1080 / image_height),
                ]
                center_point = calculate_center_point(resized_gt_box)
                image_filename = format_image_filename(coco_annotation_file, img_id)
                image_candidates.append({
                    "image_id": image_filename,
                    "category": cat_name,
                    "object_size": object_size,
                    "gt_box": resized_gt_box,
                    "point": center_point,
                    "_area_ratio": area_ratio,
                })

        if not image_candidates:
            continue

        # Keep only top-2 by area ratio per image
        image_candidates.sort(key=lambda d: d["_area_ratio"], reverse=True)
        top2 = image_candidates[:2]
        for rec in top2:
            rec.pop("_area_ratio", None)
        results.extend(top2)

    # Save multi-object results
    with open(output_file_multi, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nMulti-object filtering completed! Saved {len(results)} instances to: {output_file_multi}")
    return results

def main():
    """Main function"""
    # File paths
    coco_annotation_file = "/home/yanhao/SSHS/AudioCOCO/instances_val2014.json"
    category_file = "/home/yanhao/SSHS/AudioCOCO/category.txt"
    output_file = "/home/yanhao/SSHS/AudioCOCO/filtered_val.json"
    output_file_multi = "/home/yanhao/SSHS/AudioCOCO/filtered_val_multi.json"
    
    # Check if files exist
    if not os.path.exists(coco_annotation_file):
        print(f"Error: COCO annotation file does not exist: {coco_annotation_file}")
        print("Please ensure instances_val2014.json file exists")
        return
    
    if not os.path.exists(category_file):
        print(f"Error: Category file does not exist: {category_file}")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Execute filtering
    try:
        filtered_instances = filter_coco_instances(coco_annotation_file, category_file, output_file)
        print(f"\nFiltering results saved to: {output_file}")
        print(f"Total filtered {len(filtered_instances)} instances that meet criteria")

        # Multi-object filtering
        filtered_multi = filter_coco_multi_instances(coco_annotation_file, category_file, output_file_multi)
        print(f"Total filtered multi-object instances: {len(filtered_multi)}")
    except Exception as e:
        print(f"Error occurred during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
