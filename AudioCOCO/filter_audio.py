import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
import csv

# 加载预训练模型
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# 设置主文件夹路径
main_folder_path = "/home/yanhao/VGG-Sound-Audios/train"
results_folder = os.path.join(main_folder_path, "filtering_results")
os.makedirs(results_folder, exist_ok=True)

# 创建结果汇总CSV文件
summary_csv_path = os.path.join(results_folder, "summary_results.csv")
csv_header = ["Category", "Original_Files", "Original_Spatial", "Original_Spectral", "Original_Semantic",
              "Semantic_80_Files", "Semantic_80_Spatial", "Semantic_80_Spectral", "Semantic_80_Semantic",
              "Spectral_65_Files", "Spectral_65_Spatial", "Spectral_65_Spectral", "Spectral_65_Semantic",
              "Spatial_50_Files", "Spatial_50_Spatial", "Spatial_50_Spectral", "Spatial_50_Semantic"]

with open(summary_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_header)

def compute_channel_consistency(audio_file):
    if audio_file.ndim < 2:
        return None
    left_channel = audio_file[0]
    right_channel = audio_file[1]
    correlation_matrix = np.corrcoef(left_channel, right_channel)
    correlation = correlation_matrix[0, 1]
    return correlation

def compute_spectral_consistency(audio_file, n_fft=2048, hop_length=512):
    stft = librosa.stft(audio_file, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    spec_embedding = np.mean(magnitude, axis=1)
    spec_embedding = np.log1p(spec_embedding)
    return spec_embedding

def compute_semantic_consistency(audio_file, processor, model):
    input_values = processor(audio_file, sampling_rate=16000, return_tensors="pt", padding=True).input_values
    with torch.no_grad():
        outputs = model(input_values)
        hidden_states = outputs.last_hidden_state
    embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def process_category_folder(category_folder):
    print(f"\n\n===== Processing category: {os.path.basename(category_folder)} =====")
    
    # 获取文件夹中的所有WAV文件
    audio_files = [os.path.join(category_folder, f) for f in os.listdir(category_folder) 
                  if f.endswith('.wav')]
    
    if len(audio_files) < 4:
        print(f"Skipping {category_folder}, insufficient files: {len(audio_files)}")
        return None
    
    # 计算所有嵌入
    semantic_embeddings = []
    spectral_embeddings = []
    spitial_embeddings = []
    valid_files = []
    
    for file in tqdm(audio_files, desc=f"Processing {os.path.basename(category_folder)}"):
        try:
            y, sr = librosa.load(file, sr=16000, mono=False)
            y_mono = librosa.to_mono(y)  # 转换为单声道
            
            se_emb = compute_semantic_consistency(y_mono, processor, model)
            sp_emb = compute_spectral_consistency(y_mono)
            similarity = compute_channel_consistency(y)
            
            semantic_embeddings.append(se_emb)
            spectral_embeddings.append(sp_emb)
            valid_files.append(file)
            
            if similarity is not None:
                spitial_embeddings.append(similarity)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    if len(valid_files) < 4:
        print(f"Skipping {category_folder}, insufficient valid files: {len(valid_files)}")
        return None
    
    # 转换为numpy数组
    semantic_embeddings = np.array(semantic_embeddings)
    spectral_embeddings = np.array(spectral_embeddings)
    spitial_embeddings = np.array(spitial_embeddings)
    
    # 计算相似度矩阵
    semantic_matrix = cosine_similarity(semantic_embeddings)
    spectral_matrix = cosine_similarity(spectral_embeddings)
    
    # 计算上三角平均值（不包括对角线）
    N = semantic_matrix.shape[0]
    top80 = max(int(N * 0.8), 1)  # 修改原来的80%为80%
    top65 = max(int(N * 0.65), 1)  # 修改原来的50%为65% 
    top50 = max(int(N * 0.5), 1)  # 保留原来的50%
    
    semantic_sum = 0.0
    spectral_sum = 0.0
    count = 0
    
    for i in range(N):
        for j in range(i+1, N):
            semantic_sum += semantic_matrix[i, j]
            spectral_sum += spectral_matrix[i, j]
            count += 1
    
    semantic_score = semantic_sum / count if count > 0 else 0
    spectral_score = spectral_sum / count if count > 0 else 0
    spitial_score = np.mean(spitial_embeddings) if len(spitial_embeddings) > 0 else 0
    
    print(f"Original Spatial Consistency: {spitial_score:.4f}")
    print(f"Original Spectral Consistency: {spectral_score:.4f}")
    print(f"Original Semantic Consistency: {semantic_score:.4f}")
    
    # 修改过滤流程，检查每次过滤后的文件数量

    # --- 第1步：基于语义一致性过滤，保留前80%的音频 ---
    semantic_avg_sim = []
    for i in range(len(semantic_matrix)):
        # 排除自相似度（对角线元素）
        sim_scores = np.concatenate([semantic_matrix[i, :i], semantic_matrix[i, i+1:]])
        semantic_avg_sim.append(np.mean(sim_scores))
    
    # 按平均相似度降序排序
    indices_semantic_sorted = np.argsort(semantic_avg_sim)[::-1]
    # 选择前80%的音频文件
    top80_indices = indices_semantic_sorted[:top80]
    
    # 获取原始列表中的相应文件
    selected_files_80 = [valid_files[i] for i in top80_indices]
    print(f"Total audio files: {len(valid_files)}")
    print(f"Keeping top 80% audio files after semantic filtering: {len(selected_files_80)}")
    
    # 检查第一次过滤后是否至少有20个文件
    if len(selected_files_80) < 20:
        print(f"\n⚠️ Warning: After semantic filtering, only {len(selected_files_80)} files remain (< 20)")
        print(f"Keeping all original {len(valid_files)} files")
        
        # 使用所有原始文件作为最终结果
        selected_files_50 = valid_files
        final_indices = list(range(len(valid_files)))
        
        # 设置所有指标为原始值
        avg_semantic_80 = semantic_score
        avg_spectral_80 = spectral_score
        avg_spitial_80 = spitial_score
        
        avg_semantic_65 = semantic_score
        avg_spectral_65 = spectral_score 
        avg_spitial_65 = spitial_score
        
        avg_semantic_50 = semantic_score
        avg_spectral_50 = spectral_score
        avg_spitial_50 = spitial_score
        
        # 跳过后续的过滤步骤
        use_original = True
    else:
        # 继续正常的过滤流程
        use_original = False
        
        # 收集相应的分数
        semantic_scores_80 = semantic_embeddings[top80_indices]
        spectral_scores_80 = spectral_embeddings[top80_indices]
        spitial_scores_80 = []
        for idx in top80_indices:
            if idx < len(spitial_embeddings):
                spitial_scores_80.append(spitial_embeddings[idx])
        spitial_scores_80 = np.array(spitial_scores_80)
        
        # 计算平均值
        semantic_matrix_80 = cosine_similarity(semantic_scores_80)
        spectral_matrix_80 = cosine_similarity(spectral_scores_80)
        
        semantic_sum_80 = 0.0
        spectral_sum_80 = 0.0
        count_80 = 0
        n_80 = len(semantic_scores_80)
        for i in range(n_80):
            for j in range(i+1, n_80):
                semantic_sum_80 += semantic_matrix_80[i, j]
                spectral_sum_80 += spectral_matrix_80[i, j]
                count_80 += 1
        
        avg_semantic_80 = semantic_sum_80 / count_80 if count_80 > 0 else 0
        avg_spectral_80 = spectral_sum_80 / count_80 if count_80 > 0 else 0
        avg_spitial_80 = np.mean(spitial_scores_80) if len(spitial_scores_80) > 0 else 0
        
        print(f"After semantic filtering - Average spatial consistency: {avg_spitial_80:.4f}")
        print(f"After semantic filtering - Average spectral consistency: {avg_spectral_80:.4f}")
        print(f"After semantic filtering - Average semantic consistency: {avg_semantic_80:.4f}")
        
        # --- 第2步：基于频谱一致性过滤，从前80%的音频中保留前65% ---
        # 计算频谱相似度矩阵
        spectral_avg_sim = []
        for i in range(len(spectral_matrix_80)):
            sim_scores = np.concatenate([spectral_matrix_80[i, :i], spectral_matrix_80[i, i+1:]])
            spectral_avg_sim.append(np.mean(sim_scores))
        
        # 按平均相似度降序排序
        indices_spectral_sorted = np.argsort(spectral_avg_sim)[::-1]
        # 从前80%中选择前65%的音频文件 (这里是整体的65%/80% ≈ 81.25%)
        top65_count = max(int(len(top80_indices) * 0.8125), 1)
        top65_indices_in_80 = indices_spectral_sorted[:top65_count]
        
        # 获取原始索引
        top65_indices = [top80_indices[i] for i in top65_indices_in_80]
        
        # 获取65%的音频文件
        selected_files_65 = [valid_files[i] for i in top65_indices]
        print(f"Keeping top 65% audio files after spectral filtering: {len(selected_files_65)}")
        
        # 检查第二次过滤后是否至少有20个文件
        if len(selected_files_65) < 20:
            print(f"\n⚠️ Warning: After spectral filtering, only {len(selected_files_65)} files remain (< 20)")
            print(f"Using results from semantic filtering instead ({len(selected_files_80)} files)")
            
            # 使用语义过滤后的文件作为最终结果
            selected_files_50 = selected_files_80
            final_indices = top80_indices
            
            # 设置后续指标等于语义过滤后的值
            avg_semantic_65 = avg_semantic_80
            avg_spectral_65 = avg_spectral_80
            avg_spitial_65 = avg_spitial_80
            
            avg_semantic_50 = avg_semantic_80
            avg_spectral_50 = avg_spectral_80
            avg_spitial_50 = avg_spitial_80
            
            # 跳过剩余的过滤步骤
            skip_last_filter = True
        else:
            skip_last_filter = False
            
            # 收集相应的分数
            semantic_scores_65 = semantic_embeddings[top65_indices]
            spectral_scores_65 = spectral_embeddings[top65_indices]
            spitial_scores_65 = []
            for idx in top65_indices:
                if idx < len(spitial_embeddings):
                    spitial_scores_65.append(spitial_embeddings[idx])
            spitial_scores_65 = np.array(spitial_scores_65)
            
            # 计算平均值
            semantic_matrix_65 = cosine_similarity(semantic_scores_65)
            spectral_matrix_65 = cosine_similarity(spectral_scores_65)
            
            semantic_sum_65 = 0.0
            spectral_sum_65 = 0.0
            count_65 = 0
            n_65 = len(semantic_scores_65)
            for i in range(n_65):
                for j in range(i+1, n_65):
                    semantic_sum_65 += semantic_matrix_65[i, j]
                    spectral_sum_65 += spectral_matrix_65[i, j]
                    count_65 += 1
            
            avg_semantic_65 = semantic_sum_65 / count_65 if count_65 > 0 else 0
            avg_spectral_65 = spectral_sum_65 / count_65 if count_65 > 0 else 0
            avg_spitial_65 = np.mean(spitial_scores_65) if len(spitial_scores_65) > 0 else 0
            
            print(f"After spectral filtering - Average spatial consistency: {avg_spitial_65:.4f}")
            print(f"After spectral filtering - Average spectral consistency: {avg_spectral_65:.4f}")
            print(f"After spectral filtering - Average semantic consistency: {avg_semantic_65:.4f}")
            
            # --- 第3步：基于声道一致性过滤 ---
            if len(spitial_scores_65) > 0 and not skip_last_filter:
                # 按声道一致性降序排序
                channel_indices_sorted = np.argsort(spitial_scores_65)[::-1]
                # 从前65%中选择前50%的音频文件 (这里是整体的50%/65% ≈ 76.9%)
                top50_count = max(int(len(top65_indices) * 0.769), 1)
                top50_indices_in_65 = channel_indices_sorted[:top50_count]
                
                # 获取原始索引
                top50_indices = [top65_indices[i] for i in top50_indices_in_65]
                
                # 获取50%的音频文件
                selected_files_50 = [valid_files[i] for i in top50_indices]
                print(f"Keeping top 50% audio files after spatial filtering: {len(selected_files_50)}")
                
                # 检查第三次过滤后是否至少有20个文件
                if len(selected_files_50) < 20:
                    print(f"\n⚠️ Warning: After spatial filtering, only {len(selected_files_50)} files remain (< 20)")
                    print(f"Using results from spectral filtering instead ({len(selected_files_65)} files)")
                    
                    # 使用频谱过滤后的文件作为最终结果
                    selected_files_50 = selected_files_65
                    final_indices = top65_indices
                    
                    # 设置最终指标等于频谱过滤后的值
                    avg_semantic_50 = avg_semantic_65
                    avg_spectral_50 = avg_spectral_65
                    avg_spitial_50 = avg_spitial_65
                else:
                    final_indices = top50_indices
                    
                    # 计算最终指标
                    semantic_scores_50 = semantic_embeddings[top50_indices]
                    spectral_scores_50 = spectral_embeddings[top50_indices]
                    spitial_scores_50 = []
                    for idx in top50_indices:
                        if idx < len(spitial_embeddings):
                            spitial_scores_50.append(spitial_embeddings[idx])
                    spitial_scores_50 = np.array(spitial_scores_50)
                    
                    # 计算平均值
                    semantic_matrix_50 = cosine_similarity(semantic_scores_50)
                    spectral_matrix_50 = cosine_similarity(spectral_scores_50)
                    
                    semantic_sum_50 = 0.0
                    spectral_sum_50 = 0.0
                    count_50 = 0
                    n_50 = len(semantic_scores_50)
                    for i in range(n_50):
                        for j in range(i+1, n_50):
                            semantic_sum_50 += semantic_matrix_50[i, j]
                            spectral_sum_50 += spectral_matrix_50[i, j]
                            count_50 += 1
                    
                    avg_semantic_50 = semantic_sum_50 / count_50 if count_50 > 0 else 0
                    avg_spectral_50 = spectral_sum_50 / count_50 if count_50 > 0 else 0
                    avg_spitial_50 = np.mean(spitial_scores_50) if len(spitial_scores_50) > 0 else 0
                    
                    print(f"After spatial filtering - Average spatial consistency: {avg_spitial_50:.4f}")
                    print(f"After spatial filtering - Average spectral consistency: {avg_spectral_50:.4f}")
                    print(f"After spatial filtering - Average semantic consistency: {avg_semantic_50:.4f}")
            else:
                print("Not enough channel data for filtering in step 3")
                # 如果没有足够的声道数据，使用频谱过滤结果
                selected_files_50 = selected_files_65
                final_indices = top65_indices
                avg_semantic_50 = avg_semantic_65
                avg_spectral_50 = avg_spectral_65
                avg_spitial_50 = avg_spitial_65

    # --- 保存结果和生成图表 ---
    
    # 创建该类别的结果文件夹
    category_name = os.path.basename(category_folder)
    category_results_folder = os.path.join(results_folder, category_name)
    os.makedirs(category_results_folder, exist_ok=True)
    
    # 生成图表
    labels = ['100% Audio', '80% Audio', '65% Audio', '50% Audio']
    channel_avgs = [spitial_score, avg_spitial_80, avg_spitial_65, avg_spitial_50]
    spectral_avgs = [spectral_score, avg_spectral_80, avg_spectral_65, avg_spectral_50]
    semantic_avgs = [semantic_score, avg_semantic_80, avg_semantic_65, avg_semantic_50]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, channel_avgs, width, label='Spatial Consistency')
    rects2 = ax.bar(x, spectral_avgs, width, label='Spectral Consistency')
    rects3 = ax.bar(x + width, semantic_avgs, width, label='Semantic Consistency')
    
    ax.set_ylabel('Average Consistency Score', fontsize=22)
    ax.set_title(f'Consistency Metrics for {category_name}', fontsize=23, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16)
    ax.legend(loc='upper left', fontsize=16)
    
    # 调整坐标轴标签的字体大小
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # 保存图表前调整布局
    plt.tight_layout()
    plt.savefig(os.path.join(category_results_folder, 'consistency_metrics.png'))
    plt.close()
    
    # 保存选定的文件到日志文件
    with open(os.path.join(category_results_folder, 'selected_files.txt'), 'w') as f:
        if len(selected_files_50) == len(valid_files):
            f.write("# All original files kept (filtered result < 20 files)\n")
        else:
            f.write("# Selected files after full filtering\n")
        
        for file in selected_files_50:
            f.write(f"{os.path.basename(file)}\n")
    
    # 将结果添加到汇总CSV文件
    with open(summary_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            category_name, 
            len(valid_files), spitial_score, spectral_score, semantic_score,
            len(selected_files_80), avg_spitial_80, avg_spectral_80, avg_semantic_80,
            len(selected_files_65), avg_spitial_65, avg_spectral_65, avg_semantic_65,
            len(selected_files_50), avg_spitial_50, avg_spectral_50, avg_semantic_50
        ])
    
    return {
        "category": category_name,
        "original_files": len(valid_files),
        "final_files": len(selected_files_50),
        "original_scores": (spitial_score, spectral_score, semantic_score),
        "final_scores": (avg_spitial_50, avg_spectral_50, avg_semantic_50),
        "selected_files": selected_files_50,
        "used_original": len(selected_files_50) == len(valid_files)  # 新增标记，表示是否使用了原始文件集
    }

# 获取所有子文件夹
subdirectories = [os.path.join(main_folder_path, d) for d in os.listdir(main_folder_path) 
                 if os.path.isdir(os.path.join(main_folder_path, d)) and d != "filtering_results"]

# 从subdirectories中去掉几个文件夹
# 这些文件夹不需要处理
excluded_folders = ['/home/yanhao/VGG-Sound-Audios/train/mouse clicking', '/home/yanhao/VGG-Sound-Audios/train/ball', '/home/yanhao/VGG-Sound-Audios/train/clock',
                    '/home/yanhao/VGG-Sound-Audios/train/zebra', '/home/yanhao/VGG-Sound-Audios/train/sheep']

for folder in excluded_folders:
    if folder in subdirectories:
        subdirectories.remove(folder)

# 处理每个子文件夹
results = []
for subdir in tqdm(subdirectories, desc="Processing categories"):
    result = process_category_folder(subdir)
    if result:
        results.append(result)

print(f"\nCompleted processing {len(results)} categories")
print(f"Results saved to {results_folder}")

# 创建汇总图表
if results:
    categories = [r["category"] for r in results]
    original_channel = [r["original_scores"][0] for r in results]
    original_spectral = [r["original_scores"][1] for r in results]
    original_semantic = [r["original_scores"][2] for r in results]
    final_channel = [r["final_scores"][0] for r in results]
    final_spectral = [r["final_scores"][1] for r in results]
    final_semantic = [r["final_scores"][2] for r in results]
    
    # 排序类别，使其更具可读性
    sorted_indices = np.argsort(original_semantic)
    categories = [categories[i] for i in sorted_indices]
    original_channel = [original_channel[i] for i in sorted_indices]
    original_spectral = [original_spectral[i] for i in sorted_indices]
    original_semantic = [original_semantic[i] for i in sorted_indices]
    final_channel = [final_channel[i] for i in sorted_indices]
    final_spectral = [final_spectral[i] for i in sorted_indices]
    final_semantic = [final_semantic[i] for i in sorted_indices]
    
    # 修改汇总图表样式
    # 绘制汇总图表
    plt.figure(figsize=(15, 12))

    plt.subplot(3, 1, 1)
    indices = np.arange(len(categories))
    plt.bar(indices - 0.2, original_channel, width=0.4, label='Original', color='lightblue')
    plt.bar(indices + 0.2, final_channel, width=0.4, label='After Filtering', color='blue')
    plt.title('Spatial Consistency Before and After Filtering', fontsize=23, fontweight='bold')
    plt.ylabel('Score', fontsize=22)
    plt.yticks(fontsize=16)
    plt.xticks([], rotation=45) 
    plt.legend(loc='upper left', fontsize=15)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.subplot(3, 1, 2)
    plt.bar(indices - 0.2, original_spectral, width=0.4, label='Original', color='lightgreen')
    plt.bar(indices + 0.2, final_spectral, width=0.4, label='After Filtering', color='green')
    plt.title('Spectral Consistency Before and After Filtering', fontsize=23, fontweight='bold')
    plt.ylabel('Score', fontsize=22)
    plt.yticks(fontsize=16)
    plt.xticks([], rotation=45)
    plt.legend(loc='upper left', fontsize=15)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.subplot(3, 1, 3)
    plt.bar(indices - 0.2, original_semantic, width=0.4, label='Original', color='lightsalmon')
    plt.bar(indices + 0.2, final_semantic, width=0.4, label='After Filtering', color='red')
    plt.title('Semantic Consistency Before and After Filtering', fontsize=23, fontweight='bold')
    plt.ylabel('Score', fontsize=22)
    plt.yticks(fontsize=16)
    plt.xticks(indices, categories, rotation=45)
    plt.legend(loc='upper left', fontsize=15)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'all_categories_comparison.png'))
    # save pdf
    plt.savefig(os.path.join(results_folder, 'all_categories_comparison.pdf'))
    plt.close()

# 创建汇总文件列表
all_filtered_files_path = os.path.join(results_folder, "all_filtered_files.txt")

# 收集所有过滤后的文件
with open(all_filtered_files_path, 'w') as f:
    f.write("# All filtered files in format: category/filename\n")
    f.write("# Total categories: {}\n".format(len(results)))
    
    # 计算过滤后的总文件数
    total_filtered_files = sum(len(r["selected_files"]) for r in results)
    f.write("# Total filtered files: {}\n\n".format(total_filtered_files))
    
    # 遍历每个类别及其过滤后的文件
    for result in results:
        category = result["category"]
        for file_path in result["selected_files"]:
            filename = os.path.basename(file_path)
            # 写入格式为 "category/filename"
            f.write(f"{category}/{filename}\n")

print(f"All filtered files list saved to {all_filtered_files_path}")

# 统计过滤后的音频文件数量
if results:
    # 初始化计数器
    total_original = 0
    total_after_semantic = 0
    total_after_spectral = 0
    total_after_channel = 0
    
    # 创建结果DataFrame便于分析
    stats_df = pd.DataFrame(columns=[
        "Category", "Original", "After Semantic", "After Spectral", "After Spatial", 
        "Semantic Ratio", "Spectral Ratio", "Spatial Ratio", "Overall Ratio"
    ])
    
    # 遍历所有处理结果
    for result in results:
        category_folder = os.path.join(main_folder_path, result["category"])
        
        # 获取每个阶段的文件数量
        orig_files = result["original_files"]
        semantic_files = int(orig_files * 0.8)  # 80%
        spectral_files = int(orig_files * 0.65)  # 65%
        channel_files = result["final_files"]  # 50%
        
        # 计算比例
        semantic_ratio = semantic_files / orig_files if orig_files > 0 else 0
        spectral_ratio = spectral_files / orig_files if orig_files > 0 else 0
        channel_ratio = channel_files / orig_files if orig_files > 0 else 0
        overall_ratio = channel_files / orig_files if orig_files > 0 else 0
        
        # 添加到汇总计数
        total_original += orig_files
        total_after_semantic += semantic_files
        total_after_spectral += spectral_files
        total_after_channel += channel_files
        
        # 添加到DataFrame
        stats_df = stats_df._append({
            "Category": result["category"],
            "Original": orig_files,
            "After Semantic": semantic_files,
            "After Spectral": spectral_files,
            "After Spatial": channel_files,
            "Semantic Ratio": semantic_ratio,
            "Spectral Ratio": spectral_ratio,
            "Spatial Ratio": channel_ratio,
            "Overall Ratio": overall_ratio
        }, ignore_index=True)
    
    # 保存详细统计信息
    stats_df.to_csv(os.path.join(results_folder, "filtering_statistics.csv"), index=False)
    
    # 打印总结
    print("\n===== 音频过滤统计 =====")
    print(f"处理的类别总数: {len(results)}")
    print(f"原始音频文件总数: {total_original}")
    print(f"语义过滤后保留文件总数: {total_after_semantic} ({total_after_semantic/total_original:.2%})")
    print(f"频谱过滤后保留文件总数: {total_after_spectral} ({total_after_spectral/total_original:.2%})")
    print(f"声道过滤后保留文件总数: {total_after_channel} ({total_after_channel/total_original:.2%})")
    
    # 创建柱状图显示总过滤结果
    stages = ["Original", "After Semantic", "After Spectral", "After Spatial"]
    counts = [total_original, total_after_semantic, total_after_spectral, total_after_channel]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(stages, counts, color=['blue', 'green', 'orange', 'red'])
    
    # 在柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=12)
    
    plt.title("Audio Files Count After Each Filtering Stage", fontsize=16)
    plt.ylabel("Number of Audio Files", fontsize=14)
    plt.xlabel("Filtering Stage", fontsize=14)
    plt.xticks(rotation=45)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "filtering_stages_summary.png"))
    plt.close()
    
    # 计算每个类别保留的百分比并创建排序表
    stats_df.sort_values(by="Overall Ratio", ascending=False, inplace=True)
    
    # 创建类别保留率的柱状图
    plt.figure(figsize=(12, 8))
    plt.bar(stats_df["Category"][:20], stats_df["Overall Ratio"][:20] * 100)
    plt.title("All Categories by Audio Retention Rate", fontsize=16)
    plt.ylabel("Retention Rate (%)", fontsize=14)
    plt.xlabel("Category", fontsize=14)
    plt.xticks(rotation=45)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "top_categories_by_retention.png"))
    plt.close()

# 为每个类别创建单独的过滤统计柱状图
if results:
    print("\n生成每个类别的过滤统计图...")
    
    for result in results:
        category_name = result["category"]
        category_results_folder = os.path.join(results_folder, category_name)
        
        # 获取该类别每个阶段的文件数量
        orig_count = result["original_files"]
        semantic_count = int(orig_count * 0.8)  # 语义过滤后(80%)
        spectral_count = int(orig_count * 0.65)  # 频谱过滤后(65%)
        spatial_count = result["final_files"]  # 空间过滤后(50%)
        
        # 创建柱状图
        stages = ["Original", "After Semantic", "After Spectral", "After Spatial"]
        counts = [orig_count, semantic_count, spectral_count, spatial_count]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(stages, counts, color=['#3274A1', '#E1812C', '#3A923A', '#C03D3E'])
        
        # 在柱子上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=14)
        
        # 设置图表标题和标签
        plt.title(f"Category '{category_name}' Filtering Stages", fontsize=18)
        plt.ylabel("Number of Audio Files", fontsize=16)
        plt.ylim(0, orig_count*1.1)  # 设置y轴上限，留出空间显示数值
        plt.xticks(rotation=15, fontsize=14)
        plt.tick_params(axis='y', which='major', labelsize=14)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(category_results_folder, "filtering_stages_count.png"))
        plt.close()
    
    # 创建所有类别音频数量对比图
    plt.figure(figsize=(15, 10))
    
    # 准备数据
    categories = [r["category"] for r in results]
    original_counts = [r["original_files"] for r in results]
    final_counts = [r["final_files"] for r in results]
    
    # 按原始文件数量排序
    sort_indices = np.argsort(original_counts)[::-1]  # 降序排列
    categories = [categories[i] for i in sort_indices]
    original_counts = [original_counts[i] for i in sort_indices]
    final_counts = [final_counts[i] for i in sort_indices]
    
    # 取前20个类别(如果有)进行展示
    display_limit = min(20, len(categories))
    categories = categories[:display_limit]
    original_counts = original_counts[:display_limit]
    final_counts = final_counts[:display_limit]
    
    # 创建柱状图
    x = np.arange(len(categories))  # x轴位置
    width = 0.35  # 柱宽度
    
    fig, ax = plt.subplots(figsize=(15, 8))
    rects1 = ax.bar(x - width/2, original_counts, width, label='Original Files', color='#3274A1')
    rects2 = ax.bar(x + width/2, final_counts, width, label='After Filtering', color='#C03D3E')
    
    # 添加标题和标签
    ax.set_title('Original vs Filtered Audio Files', fontsize=23, fontweight='bold')
    ax.set_ylabel('Number of Files', fontsize=22)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=16)
    ax.legend(fontsize=16)
    ax.tick_params(axis='y', which='major', labelsize=16)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # 调整布局
    fig.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(results_folder, "category_files_comparison.png"))
    plt.savefig(os.path.join(results_folder, "category_files_comparison.pdf"))
    plt.close()
    
    # 计算并可视化过滤保留率
    retention_rates = [r["final_files"]/r["original_files"]*100 for r in results]
    
    # 按保留率排序
    sort_indices = np.argsort(retention_rates)[::-1]  # 降序排列
    sorted_categories = [categories[i] for i in sort_indices if i < len(categories)]
    sorted_rates = [retention_rates[i] for i in sort_indices if i < len(categories)]
    
    # 取前20个展示
    display_limit = min(20, len(sorted_categories))
    sorted_categories = sorted_categories[:display_limit]
    sorted_rates = sorted_rates[:display_limit]
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(sorted_categories, sorted_rates, color='#3A923A')
    
    # 在柱子上添加百分比标签
    for bar, rate in zip(bars, sorted_rates):
        plt.text(bar.get_x() + bar.get_width()/2., rate + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, rotation=0)
    
    plt.title('Categories by Retention Rate', fontsize=18)
    plt.ylabel('Retention Rate (%)', fontsize=16)
    plt.ylim(0, max(sorted_rates)*1.1)  # 设置y轴上限
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tight_layout()
    
    plt.savefig(os.path.join(results_folder, "category_retention_rates.png"))
    plt.close()
    
    print(f"已生成所有类别的过滤统计图，保存在各自的结果文件夹中")