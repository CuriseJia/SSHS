import os
import numpy as np
from collections import defaultdict

def is_point_in_box(x, y, box):
    """Determine if the point (x,y) is in the box, the box format is [center x, center y, width, height]"""
    center_x, center_y, width, height = box
    half_width = width / 2
    half_height = height / 2
    return (center_x - half_width <= x <= center_x + half_width) and (center_y - half_height <= y <= center_y + half_height)

def calculate_center_distance(x, y, box):
    """Calculate the distance from the point (x,y) to the center of the box, the box format is [center x, center y, width, height]"""
    center_x, center_y, _, _ = box
    return np.sqrt((x - center_x)**2 + (y - center_y)**2)

def calculate_x_distance(x, box):
    """Calculate the distance from the x coordinate of the point to the x coordinate of the center of the box"""
    center_x, _, _, _ = box
    return abs(x - center_x)

def calculate_y_distance(y, box):
    """Calculate the distance from the y coordinate of the point to the y coordinate of the center of the box"""
    _, center_y, _, _ = box
    return abs(y - center_y)

def distance_to_degree(distance):
    """Convert distance to angle (degrees)"""
    return distance / 960 * 25

def read_data_from_file(file_path):
    """Read data from file"""
    data = []
    with open(file_path, 'r') as f:
        # Skip header line
        header = f.readline().strip()
        
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 11:
                try:
                    record = {
                        'x': float(parts[0]),
                        'y': float(parts[1]),
                        'image': parts[2],
                        'audio': parts[3],
                        'category': parts[4],
                        'object_size': parts[5],
                        'gt_box': [float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])],
                        'task': parts[10],
                        'time': float(parts[11]) if len(parts) > 11 else 0.0
                    }
                    data.append(record)
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line.strip()} - {e}")
                    continue
    return data

def analyze_distance_by_degrees(file_path, degree_step=1.5):
    """Analyze distance data by angle intervals"""
    records = read_data_from_file(file_path)
    
    # Initialize data by angle intervals
    degree_bins = np.arange(0, 25 + degree_step, degree_step)
    data_by_degree = defaultdict(list)
    x_distance_by_degree = defaultdict(list)
    y_distance_by_degree = defaultdict(list)
    
    for record in records:
        x = record['x']
        y = record['y']
        gt_box = record['gt_box']
        
        
        x_distance = calculate_x_distance(x, gt_box)
        y_distance = calculate_y_distance(y, gt_box)
        
        x_degree = distance_to_degree(x_distance)
        y_degree = distance_to_degree(y_distance)
        
        in_box = is_point_in_box(x, y, gt_box)
        
        record_info = {
            'x': x,
            'y': y,
            'x_distance': x_distance,
            'x_degree': x_degree,
            'y_distance': y_distance,
            'y_degree': y_degree,
            'in_box': in_box,
            'task': record['task'],
            'category': record['category'],
            'object_size': record['object_size'],
            'image': record['image'],
            'audio': record['audio']
        }
        
        # Group by X distance angle
        for i in range(len(degree_bins) - 1):
            if degree_bins[i] <= x_degree < degree_bins[i+1]:
                bin_key = f"{degree_bins[i]:.1f}-{degree_bins[i+1]:.1f}"
                x_distance_by_degree[bin_key].append(record_info)
                break
        
        # Group by Y distance angle
        for i in range(len(degree_bins) - 1):
            if degree_bins[i] <= y_degree < degree_bins[i+1]:
                bin_key = f"{degree_bins[i]:.1f}-{degree_bins[i+1]:.1f}"
                y_distance_by_degree[bin_key].append(record_info)
                break
    
    return x_distance_by_degree, y_distance_by_degree

def print_statistics(data_dict, title):
    """Print statistics information"""
    print(f"\n===== {title} =====")
    
    # Calculate the total number of all records
    total_records = sum(len(records) for records in data_dict.values())
    
    print("Angle interval\tData count\tData ratio\tClick accuracy\tTask2 count\tTask2 accuracy\tTask4 count\tTask4 accuracy")
    
    for bin_key, records in sorted(data_dict.items()):
        if records:
            # Calculate the overall accuracy
            total = len(records)
            data_ratio = total / total_records if total_records > 0 else 0
            correct = sum(1 for r in records if r['in_box'])
            accuracy = correct / total if total > 0 else 0
            
            # Calculate the accuracy of Task2
            task2_records = [r for r in records if r['task'] == '2']
            task2_total = len(task2_records)
            task2_correct = sum(1 for r in task2_records if r['in_box'])
            task2_accuracy = task2_correct / task2_total if task2_total > 0 else 0
            
            # Calculate the accuracy of Task4
            task4_records = [r for r in records if r['task'] == '4']
            task4_total = len(task4_records)
            task4_correct = sum(1 for r in task4_records if r['in_box'])
            task4_accuracy = task4_correct / task4_total if task4_total > 0 else 0
            
            print(f"{bin_key}\t{total}\t{data_ratio:.2%}\t{accuracy:.4f}\t{task2_total}\t{task2_accuracy:.4f}\t{task4_total}\t{task4_accuracy:.4f}")


def write_detailed_data(data_dict, output_dir, filename_prefix):
    """Write the detailed data of each angle interval to the file"""
    os.makedirs(output_dir, exist_ok=True)
    
    for bin_key, records in sorted(data_dict.items()):
        if records:
            # Replace special characters,使之适合作为文件名
            safe_bin_key = bin_key.replace('.', '_').replace('-', 'to')
            filename = os.path.join(output_dir, f"{filename_prefix}_{safe_bin_key}.txt")
            
            with open(filename, 'w') as f:
                # Write the header line
                f.write("x\ty\tx_distance\tx_degree\ty_distance\ty_degree\tin_box\ttask\tcategory\tobject_size\timage\taudio\n")
                
                # Write each record
                for r in records:
                    f.write(f"{r['x']}\t{r['y']}\t"
                            f"{r['x_distance']:.2f}\t{r['x_degree']:.2f}\t{r['y_distance']:.2f}\t{r['y_degree']:.2f}\t"
                            f"{1 if r['in_box'] else 0}\t{r['task']}\t{r['category']}\t{r['object_size']}\t"
                            f"{r['image']}\t{r['audio']}\n")
            
            print(f"Saved {len(records)} records to file: {filename}")

def main():
    # Set the input file path
    file_path = "/home/yanhao/SSHS/result/result_all.txt"
    output_dir = "/home/yanhao/SSHS/distance_analysis"
    
    # Analyze data
    x_distance_by_degree, y_distance_by_degree = analyze_distance_by_degrees(file_path)
    
    # Print statistics information
    print_statistics(x_distance_by_degree, "Statistics by X direction distance angle")
    print_statistics(y_distance_by_degree, "Statistics by Y direction distance angle")
    
    # Write the detailed data to the file
    # write_detailed_data(data_by_degree, output_dir, "center_degree")
    write_detailed_data(x_distance_by_degree, output_dir, "x_degree")
    write_detailed_data(y_distance_by_degree, output_dir, "y_degree")

if __name__ == "__main__":
    main()
