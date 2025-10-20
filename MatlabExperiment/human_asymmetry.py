import os
import glob
import random
import math
import pandas as pd
from collections import defaultdict

def read_txt_files(directory_path):
    """
    read all txt files in the directory
    data format: x	y	image	audio	category	object_size	gt_box	task	time
    """
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
    all_data = []
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line and i > 0:
                        # x	y	image	audio	category	object_size	gt_box	task	time
                        parts = line.split('\t')
                        if len(parts) >= 12:
                            try:
                                x = float(parts[0])
                                y = float(parts[1])
                                image = parts[2]
                                audio = parts[3]
                                category = parts[4]
                                object_size = parts[5]
                                gt_box_x = float(parts[6])
                                gt_box_y = float(parts[7])
                                gt_box_w = float(parts[8])
                                gt_box_h = float(parts[9])
                                task = int(parts[10])
                                time = float(parts[11])
                                
                                all_data.append({
                                    'x': x,
                                    'y': y,
                                    'image': image,
                                    'audio': audio,
                                    'category': category,
                                    'object_size': object_size,
                                    'gt_box': {
                                        'x': gt_box_x,
                                        'y': gt_box_y,
                                        'w': gt_box_w,
                                        'h': gt_box_h
                                    },
                                    'task': task,
                                    'time': time,
                                    'file': os.path.basename(file_path)
                                })
                            except (ValueError, IndexError) as e:
                                print(f"Error parsing line data: {line[:50]}... Error: {e}")
                                continue
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    return all_data

def filter_task_4(data):
    return [item for item in data if item['task'] == 4]

def random_sampling(data, sample_ratio=0.7):
    random.shuffle(data)
    sample_size = int(len(data) * sample_ratio)
    return data[:sample_size]

def calculate_angles_to_gt_center(x, y, gt_box, max_angle=25.6):
    # calculate the angle between x and y direction to the gt_box center
    # gt_box format: [x, y, w, h]
    # (1920, 1080) corresponds to 25.6 degrees
    gt_center_x = gt_box['x'] + gt_box['w'] / 2
    gt_center_y = gt_box['y'] + gt_box['h'] / 2
    
    # calculate the offset of x and y direction to the gt_box center
    dx = x - gt_center_x
    dy = y - gt_center_y
    
    angle_x = abs(dx / 1920) * max_angle
    angle_y = abs(dy / 1080) * max_angle
    
    return angle_x, angle_y

def analyze_by_object_size(data):
    """
    group by object_size, analyze the ratio of x and y direction within 6 degrees and outside 6 degrees
    """
    results = defaultdict(lambda: {
        'total': 0,
        'x_within_6deg': 0, 'x_beyond_6deg': 0,
        'y_within_6deg': 0, 'y_beyond_6deg': 0
    })
    
    for item in data:
        x, y = item['x'], item['y']
        object_size = item['object_size']
        gt_box = item['gt_box']
        
        angle_x, angle_y = calculate_angles_to_gt_center(x, y, gt_box)
        
        within_6deg_x = angle_x <= 6.0
        within_6deg_y = angle_y <= 6.0
        
        results[object_size]['total'] += 1
        
        if within_6deg_x:
            results[object_size]['x_within_6deg'] += 1
        else:
            results[object_size]['x_beyond_6deg'] += 1
            
        if within_6deg_y:
            results[object_size]['y_within_6deg'] += 1
        else:
            results[object_size]['y_beyond_6deg'] += 1
    
    return results

def print_results(results):
    print("=" * 80)
    print("Analysis results by object_size (x and y direction)")
    print("=" * 80)
    
    for object_size, stats in results.items():
        total = stats['total']
        x_within = stats['x_within_6deg']
        x_beyond = stats['x_beyond_6deg']
        y_within = stats['y_within_6deg']
        y_beyond = stats['y_beyond_6deg']
        
        if total > 0:
            x_within_ratio = (x_within / total) * 100
            x_beyond_ratio = (x_beyond / total) * 100
            y_within_ratio = (y_within / total) * 100
            y_beyond_ratio = (y_beyond / total) * 100
            
            print(f"\nObject Size: {object_size}")
            print(f"  Total samples: {total}")
            print(f"  X direction:")
            print(f"    Within 6 degrees: {x_within} ({x_within_ratio:.2f}%)")
            print(f"    Beyond 6 degrees: {x_beyond} ({x_beyond_ratio:.2f}%)")
            print(f"  Y direction:")
            print(f"    Within 6 degrees: {y_within} ({y_within_ratio:.2f}%)")
            print(f"    Beyond 6 degrees: {y_beyond} ({y_beyond_ratio:.2f}%)")

def main():
    directory_path = "/home/yanhao/SSHS/MatlabExperiment"
    
    print("Reading txt files...")
    all_data = read_txt_files(directory_path)
    print(f"Total {len(all_data)} data read")
    
    if len(all_data) == 0:
        print("No data read!")
        return
    
    print("Filtering task4 samples...")
    task_4_data = filter_task_4(all_data)
    print(f"Total {len(task_4_data)} task4 samples")
    
    if len(task_4_data) == 0:
        print("No task4 samples found!")
        print("Available task values:", sorted(set(item['task'] for item in all_data)))
        return
    
    print("Random sampling 70% of data...")
    sampled_data = random_sampling(task_4_data, 0.7)
    print(f"Total {len(sampled_data)} sampled data")
    
    print("Analyzing data...")
    results = analyze_by_object_size(sampled_data)
    
    print_results(results)

if __name__ == "__main__":
    main()
