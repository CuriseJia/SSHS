import json
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from collections import defaultdict, Counter
from scipy.ndimage import gaussian_filter

def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def filter_by_category_limit(instances, max_count=150):
    """Limit each category and size to at most max_count instances, and remove specified categories"""
    # Filter out all unwanted categories at once
    excluded_categories = ['sheep', 'clock', 'ball', 'mouse clicking', 'zebra']
    filtered_instances = [
        instance for instance in instances 
        if instance['category'].lower() not in excluded_categories
    ]
    
    print(f"Removed {len(instances) - len(filtered_instances)} instances from excluded categories")
    
    # Group by category and object_size
    category_size_instances = defaultdict(lambda: defaultdict(list))
    for instance in filtered_instances:
        # Use existing object_size field
        object_size = instance.get('object_size', 'size1')  # If no object_size field, default to size1
        category_size_instances[instance['category']][object_size].append(instance)
    
    # Limit quantity for each category and size
    result_instances = []
    for category, size_dict in category_size_instances.items():
        print(f"Category: {category}")
        for size, instances_list in size_dict.items():
            print(f"  - Size: {size}, Original count: {len(instances_list)}")
            if len(instances_list) > max_count:
                # Randomly select max_count instances
                selected = random.sample(instances_list, max_count)
                result_instances.extend(selected)
                print(f"    - Limited to {max_count} instances")
            else:
                result_instances.extend(instances_list)
    
    return result_instances

def analyze_spatial_distribution(instances, grid_size=10):
    """Generate virtual spatial distribution for instances and analyze uniformity"""
    # Assign random positions to each instance
    for instance in instances:
        if 'x' not in instance or 'y' not in instance:
            instance['x'] = random.random()
            instance['y'] = random.random()
    
    # Group by category
    category_instances = defaultdict(list)
    for instance in instances:
        category_instances[instance['category']].append(instance)
    
    # Analyze spatial distribution for each category
    filtered_results = []
    for category, instances_list in category_instances.items():
        # Calculate instances in the center and periphery regions
        center_instances = []
        periphery_instances = []
        
        for instance in instances_list:
            x, y = instance['x'], instance['y']
            # Define center region (40% of the middle region)
            if 0.3 <= x <= 0.7 and 0.3 <= y <= 0.7:
                center_instances.append(instance)
            else:
                periphery_instances.append(instance)
        
        # Calculate the ratio of center region
        total_count = len(instances_list)
        center_count = len(center_instances)
        center_ratio = center_count / total_count if total_count > 0 else 0
        
        print(f"Category: {category}, Center ratio: {center_ratio:.2f}")
        
        # If the center region ratio is too high (more than 50%), remove some center instances randomly
        if center_ratio > 0.5:
            # Calculate the expected number of center region instances
            target_center_count = int(total_count * 0.4)  # Expected center region占40%
            excess_count = center_count - target_center_count
            
            if excess_count > 0:
                # Randomly select the center instances to keep
                center_instances = random.sample(center_instances, center_count - excess_count)
                print(f"  - Removed {excess_count} instances from center region")
        
        # Merge results
        filtered_results.extend(periphery_instances)
        filtered_results.extend(center_instances)
    
    return filtered_results

def plot_spatial_distribution(instances, object_size, output_path, size_filter=None):
    """
    Plot spatial distribution heatmap
    
    Args:
    instances - Instance list
    object_size - Object size description(Single/Multiple)
    output_path - Output file path
    size_filter - If specified, only plot instances with this size
    """
    # If size filter is specified, only keep instances with this size
    if size_filter:
        instances = [instance for instance in instances if instance.get('object_size') == size_filter]
        print(f"Filtered to {len(instances)} instances with size '{size_filter}'")
    
    # Group by category
    category_instances = defaultdict(list)
    for instance in instances:
        category_instances[instance['category']].append(instance)
    
    # Determine subplot layout
    categories = sorted(category_instances.keys())
    n_categories = len(categories)
    
    # If no data, display an empty plot and return
    if n_categories == 0:
        plt.figure(figsize=(8, 6), facecolor='white')
        plt.text(0.5, 0.5, f"No data for {object_size} - {size_filter}", 
                 ha='center', va='center', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Empty plot saved to: {output_path} (no matching data)")
        return
    
    n_cols = min(4, n_categories)
    n_rows = (n_categories + n_cols - 1) // n_cols
    
    # Create a plot with white background
    plt.figure(figsize=(n_cols * 4, n_rows * 3.5), facecolor='white')
    
    # Set global title, font size increased
    title = f"Spatial Distribution for All Categories - Object {object_size}"
    if size_filter:
        title += f" - {size_filter}"
    plt.suptitle(title, fontsize=23, y=0.98, fontweight='bold')
    
    # Create custom color mapping - changed to red
    cmap = plt.cm.Reds
    
    for i, category in enumerate(categories):
        instances_list = category_instances[category]
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # Extract coordinates
        x_coords = [instance['x'] for instance in instances_list]
        y_coords = [instance['y'] for instance in instances_list]
        
        # Use heatmap style (no smoothing)
        if len(x_coords) > 0:
            # Use 2D histogram, no smoothing
            heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, 
                                                   bins=20, range=[[0, 1], [0, 1]])
            extent = [0, 1, 0, 1]
            
            # Directly plot heatmap, no smoothing
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', 
                         cmap=cmap, aspect='auto', interpolation='nearest')
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=14)
        
        # Set subplot title and style, font size increased
        ax.set_title(category, fontsize=22, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=20)
        ax.set_ylabel('Y Coordinate', fontsize=20)
        # Set subplot ticks, font size increased
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    # Add shared color bar, font size increased
    cax = plt.axes([0.92, 0.15, 0.02, 0.7])  # Right color bar position
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Object Count', fontsize=22)
    cbar.ax.tick_params(labelsize=18)  # Increase color bar tick font size
    
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])  # Leave space for color bar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # Save as pdf format
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Spatial distribution heatmap saved to: {output_path}")

def plot_category_statistics(instances, object_size, output_path):
    """Plot category statistics histogram, display three sizes for each category, remove numeric labels"""
    # Count instances for each category and size
    category_size_counts = defaultdict(lambda: defaultdict(int))
    
    # Count instances for each category and size
    for instance in instances:
        category = instance['category']
        size = instance.get('object_size', 'size1')  # Use object_size instead of size field
        category_size_counts[category][size] += 1
    
    # Sort categories
    categories = sorted(category_size_counts.keys())
    
    # Set size labels and use stronger contrast colors
    size_ranges = ["Size1 (0-5%)", "Size2 (5-15%)", "Size3 (15-30%)"]
    size_labels = ["size1", "size2", "size3"]
    # Use stronger contrast colors instead of original blue gradient
    size_colors = ['#3498db', '#2ecc71', '#e74c3c']  # 蓝色、绿色、红色
    
    # Create x-axis position
    x = np.arange(len(categories))
    width = 0.25  # 柱的宽度
    
    # Create plot
    plt.figure(figsize=(14, 8), facecolor='white')
    
    # Plot histogram for each size
    bars = []
    for i, size in enumerate(size_labels):
        counts = [category_size_counts[cat].get(size, 0) for cat in categories]
        bar = plt.bar(x + (i-1)*width, counts, width, label=size_ranges[i], 
                      color=size_colors[i], edgecolor='#333333')
        bars.append(bar)
    
    # Set x-axis ticks and labels
    plt.xticks(x, categories, rotation=45, ha='right', fontsize=18)
    plt.ylabel('Instance Count', fontsize=22)
    plt.yticks(fontsize=18)
    
    # Add title
    plt.title(f"{object_size} Object Distribution by Category", fontsize=23, fontweight='bold')
    
    # Add legend, font size increased two sizes
    plt.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1, 1.1))
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.gca().set_axisbelow(True)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Category statistics chart saved to: {output_path}")


def map_size_format(instances):
    """Map object_size field from percentage format to size1/size2/size3 format"""
    size_mapping = {
        "0-5%": "size1",
        "5-15%": "size2", 
        "15-30%": "size3"
    }
    
    for instance in instances:
        if 'object_size' in instance:
            instance['object_size'] = size_mapping.get(instance['object_size'], "size1")
    
    return instances

def main():
    output_dir = "/home/yanhao/SSHS/AudioCOCO/"
    os.makedirs(output_dir, exist_ok=True)
    
    single_object_path = "/home/yanhao/SSHS/AudioCOCO/filtered_val.json"
    single_objects = load_json(single_object_path)
    print(f"\nProcessing single object instances: {len(single_objects)} total")
    
    single_objects = map_size_format(single_objects)
    
    filtered_single = filter_by_category_limit(single_objects)
    
    final_single = analyze_spatial_distribution(filtered_single)
    
    filtered_single_path = os.path.join(output_dir, "final_val.json")
    with open(filtered_single_path, 'w') as f:
        json.dump(final_single, f, indent=2)
    print(f"Filtered single objects saved: {len(final_single)} instances")
    
    size_labels = ["size1", "size2", "size3"]
    for size in size_labels:
        plot_path = os.path.join(output_dir, f"single_object_distribution_{size}.png")
        plot_spatial_distribution(final_single, "Single", plot_path, size_filter=size)
    
    plot_path = os.path.join(output_dir, "single_object_distribution_all.png")
    plot_spatial_distribution(final_single, "Single", plot_path)
    
    stats_path = os.path.join(output_dir, "single_object_statistics.png")
    plot_category_statistics(final_single, "Single", stats_path)
    
    multi_object_path = "/home/yanhao/SSHS/AudioCOCO/filtered_val_multi.json"
    multi_objects = load_json(multi_object_path)
    print(f"\nProcessing multi object instances: {len(multi_objects)} total")
    
    multi_objects = map_size_format(multi_objects)
    
    filtered_multi = filter_by_category_limit(multi_objects)
    final_multi = analyze_spatial_distribution(filtered_multi)
    
    filtered_multi_path = os.path.join(output_dir, "final_val_multi.json")
    with open(filtered_multi_path, 'w') as f:
        json.dump(final_multi, f, indent=2)
    print(f"Filtered multi objects saved: {len(final_multi)} instances")
    
    for size in size_labels:
        plot_path = os.path.join(output_dir, f"multi_object_distribution_{size}.png")
        plot_spatial_distribution(final_multi, "Multiple", plot_path, size_filter=size)
    
    plot_path = os.path.join(output_dir, "multi_object_distribution_all.png")
    plot_spatial_distribution(final_multi, "Multiple", plot_path)
    
    stats_path = os.path.join(output_dir, "multi_object_statistics.png")
    plot_category_statistics(final_multi, "Multiple", stats_path)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()