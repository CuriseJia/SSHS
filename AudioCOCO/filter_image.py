import json
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from collections import defaultdict, Counter
from scipy.ndimage import gaussian_filter

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def filter_by_category_limit(instances, max_count=150):
    """限制每个类别每个尺寸最多保留max_count个实例，并移除指定类别"""
    # 一次性过滤掉所有不需要的类别
    excluded_categories = ['sheep', 'clock', 'ball', 'mouse clicking', 'zebra']
    filtered_instances = [
        instance for instance in instances 
        if instance['category'].lower() not in excluded_categories
    ]
    
    print(f"Removed {len(instances) - len(filtered_instances)} instances from excluded categories")
    
    # 按类别和object_size分组
    category_size_instances = defaultdict(lambda: defaultdict(list))
    for instance in filtered_instances:
        # 使用已有的object_size字段
        object_size = instance.get('object_size', 'size1')  # 如果没有object_size字段，默认为size1
        category_size_instances[instance['category']][object_size].append(instance)
    
    # 对每个类别的每个尺寸限制数量
    result_instances = []
    for category, size_dict in category_size_instances.items():
        print(f"Category: {category}")
        for size, instances_list in size_dict.items():
            print(f"  - Size: {size}, Original count: {len(instances_list)}")
            if len(instances_list) > max_count:
                # 随机选择max_count个实例
                selected = random.sample(instances_list, max_count)
                result_instances.extend(selected)
                print(f"    - Limited to {max_count} instances")
            else:
                result_instances.extend(instances_list)
    
    return result_instances

def analyze_spatial_distribution(instances, grid_size=10):
    """为实例生成虚拟空间分布并分析均匀性"""
    # 为每个实例分配随机位置
    for instance in instances:
        if 'x' not in instance or 'y' not in instance:
            instance['x'] = random.random()
            instance['y'] = random.random()
    
    # 按类别分组
    category_instances = defaultdict(list)
    for instance in instances:
        category_instances[instance['category']].append(instance)
    
    # 对每个类别分析空间分布
    filtered_results = []
    for category, instances_list in category_instances.items():
        # 计算中心区域和外围区域的实例
        center_instances = []
        periphery_instances = []
        
        for instance in instances_list:
            x, y = instance['x'], instance['y']
            # 定义中心区域（中间40%的区域）
            if 0.3 <= x <= 0.7 and 0.3 <= y <= 0.7:
                center_instances.append(instance)
            else:
                periphery_instances.append(instance)
        
        # 计算中心区域占比
        total_count = len(instances_list)
        center_count = len(center_instances)
        center_ratio = center_count / total_count if total_count > 0 else 0
        
        print(f"Category: {category}, Center ratio: {center_ratio:.2f}")
        
        # 如果中心区域占比过高（超过50%），则随机移除一部分中心实例
        if center_ratio > 0.5:
            # 计算期望的中心区域实例数量
            target_center_count = int(total_count * 0.4)  # 希望中心区域占40%
            excess_count = center_count - target_center_count
            
            if excess_count > 0:
                # 随机选择要保留的中心实例
                center_instances = random.sample(center_instances, center_count - excess_count)
                print(f"  - Removed {excess_count} instances from center region")
        
        # 合并结果
        filtered_results.extend(periphery_instances)
        filtered_results.extend(center_instances)
    
    return filtered_results

def plot_spatial_distribution(instances, object_size, output_path, size_filter=None):
    """
    绘制空间分布热力图
    
    参数:
    instances - 实例列表
    object_size - 对象尺寸描述(Single/Multiple)
    output_path - 输出文件路径
    size_filter - 如果指定，则只绘制该尺寸的实例
    """
    # 如果指定了尺寸过滤器，则只保留该尺寸的实例
    if size_filter:
        instances = [instance for instance in instances if instance.get('object_size') == size_filter]
        print(f"Filtered to {len(instances)} instances with size '{size_filter}'")
    
    # 按类别分组
    category_instances = defaultdict(list)
    for instance in instances:
        category_instances[instance['category']].append(instance)
    
    # 确定子图布局
    categories = sorted(category_instances.keys())
    n_categories = len(categories)
    
    # 如果没有数据，显示一个空图并返回
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
    
    # 创建具有白色背景的图形
    plt.figure(figsize=(n_cols * 4, n_rows * 3.5), facecolor='white')
    
    # 设置全局标题，字号增大
    title = f"Spatial Distribution for All Categories - Object {object_size}"
    if size_filter:
        title += f" - {size_filter}"
    plt.suptitle(title, fontsize=23, y=0.98, fontweight='bold')
    
    # 创建自定义颜色映射 - 改为红色系
    cmap = plt.cm.Reds
    
    for i, category in enumerate(categories):
        instances_list = category_instances[category]
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # 提取坐标
        x_coords = [instance['x'] for instance in instances_list]
        y_coords = [instance['y'] for instance in instances_list]
        
        # 使用热力图样式（无平滑处理）
        if len(x_coords) > 0:
            # 使用 2D 直方图，不应用平滑
            heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, 
                                                   bins=20, range=[[0, 1], [0, 1]])
            extent = [0, 1, 0, 1]
            
            # 直接绘制热图，无平滑处理
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', 
                         cmap=cmap, aspect='auto', interpolation='nearest')
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=14)
        
        # 设置子图标题和样式，字号增大
        ax.set_title(category, fontsize=22, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=20)
        ax.set_ylabel('Y Coordinate', fontsize=20)
        # 设置子图刻度,字号增大
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    # 添加共享颜色条，字号增大
    cax = plt.axes([0.92, 0.15, 0.02, 0.7])  # 右侧颜色条位置
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Object Count', fontsize=22)
    cbar.ax.tick_params(labelsize=18)  # 增大颜色条刻度字号
    
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])  # 为颜色条留出空间
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # 保存为pdf格式
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Spatial distribution heatmap saved to: {output_path}")

def plot_category_statistics(instances, object_size, output_path):
    """绘制类别统计柱状图，每个category的三个size分组显示，去除数字标注"""
    # 统计每个类别每个尺寸的实例数量
    category_size_counts = defaultdict(lambda: defaultdict(int))
    
    # 计算每个类别每个尺寸的计数
    for instance in instances:
        category = instance['category']
        size = instance.get('object_size', 'size1')  # 使用object_size而不是size字段
        category_size_counts[category][size] += 1
    
    # 排序类别
    categories = sorted(category_size_counts.keys())
    
    # 设置尺寸标签和使用更强烈的对比色
    size_ranges = ["Size1 (0-5%)", "Size2 (5-15%)", "Size3 (15-30%)"]
    size_labels = ["size1", "size2", "size3"]
    # 使用更强烈的对比色代替原来的蓝色渐变
    size_colors = ['#3498db', '#2ecc71', '#e74c3c']  # 蓝色、绿色、红色
    
    # 创建x轴位置
    x = np.arange(len(categories))
    width = 0.25  # 柱的宽度
    
    # 创建图形
    plt.figure(figsize=(14, 8), facecolor='white')
    
    # 绘制每个尺寸的柱状图
    bars = []
    for i, size in enumerate(size_labels):
        counts = [category_size_counts[cat].get(size, 0) for cat in categories]
        bar = plt.bar(x + (i-1)*width, counts, width, label=size_ranges[i], 
                      color=size_colors[i], edgecolor='#333333')
        bars.append(bar)
    
    # 设置x轴刻度和标签
    plt.xticks(x, categories, rotation=45, ha='right', fontsize=18)
    plt.ylabel('Instance Count', fontsize=22)
    plt.yticks(fontsize=18)
    
    # 添加标题
    plt.title(f"{object_size} Object Distribution by Category", fontsize=23, fontweight='bold')
    
    # 添加图例，字号增大两号
    plt.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1, 1.1))
    
    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.gca().set_axisbelow(True)  # 确保网格线在柱子下方
    # 隐藏右侧和上测的边框
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Category statistics chart saved to: {output_path}")
def map_size_format(instances):
    """将 object_size 字段从百分比格式转换为 size1/size2/size3 格式"""
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
    output_dir = "/home/yanhao/AudioCOCO/filtered_output_train"
    os.makedirs(output_dir, exist_ok=True)
    
    single_object_path = "/home/yanhao/AudioCOCO/output/single_object_instances.json"
    single_objects = load_json(single_object_path)
    print(f"\nProcessing single object instances: {len(single_objects)} total")
    
    single_objects = map_size_format(single_objects)
    
    filtered_single = filter_by_category_limit(single_objects)
    
    final_single = analyze_spatial_distribution(filtered_single)
    
    filtered_single_path = os.path.join(output_dir, "filtered_single_object_instances.json")
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
    
    multi_object_path = "/home/yanhao/AudioCOCO/output/multi_object_instances.json"
    multi_objects = load_json(multi_object_path)
    print(f"\nProcessing multi object instances: {len(multi_objects)} total")
    
    multi_objects = map_size_format(multi_objects)
    
    filtered_multi = filter_by_category_limit(multi_objects)
    final_multi = analyze_spatial_distribution(filtered_multi)
    
    filtered_multi_path = os.path.join(output_dir, "filtered_multi_object_instances.json")
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