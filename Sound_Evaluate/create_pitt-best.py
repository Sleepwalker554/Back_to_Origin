"""
创建 Pitt-best 数据集
从四个数据集（raw, mossformer, frcrn_se, demucs）中选择每个文件 ovrl 分数最高的版本
保持 dementia 和 control 的文件夹结构
"""

import pandas as pd
import os
import shutil
import csv
from pathlib import Path


def create_pitt_best_dataset(
    csv_path: str,
    output_dir: str = None
):
    """
    创建 Pitt-best 数据集
    
    参数:
        csv_path: audio_quality_evaluation.csv 文件路径
        output_dir: 输出目录路径，默认为 ad_detection/data/denoised/Pitt-Best
    """
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 设置默认输出路径
    if output_dir is None:
        output_dir = "../ad_detection/data/denoised/Pitt-Best"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建 Dementia 和 Control 子文件夹（注意首字母大写）
    dementia_dir = os.path.join(output_dir, "Dementia")
    control_dir = os.path.join(output_dir, "Control")
    os.makedirs(dementia_dir, exist_ok=True)
    os.makedirs(control_dir, exist_ok=True)
    
    # 统计信息
    stats = {
        'raw': 0,
        'mossformer': 0,
        'frcrn_se': 0,
        'demucs': 0,
        'total': 0,
        'missing': 0
    }
    
    # 定义数据集映射（使用相对路径，从 Noise_Remove 文件夹）
    dataset_mapping = {
        'raw': '../ad_detection/data/raw/Pitt',
        'mossformer': '../ad_detection/data/denoised/Pitt-MossFormer',
        'frcrn_se': '../ad_detection/data/denoised/Pitt-FRCRN_SE',
        'demucs': '../ad_detection/data/denoised/Pitt-Demucs'
    }
    
    # 处理每一行并收集详细信息
    print("\n开始处理文件...")
    selection_details = []
    
    for idx, row in df.iterrows():
        file_name = row['file_name']
        category = row['category']  # 'Control' 或 'Dementia'
        
        # 获取四个数据集的 ovrl 分数
        raw_ovrl = row['raw_ovrl']
        mossformer_ovrl = row['mossformer_ovrl']
        frcrn_se_ovrl = row['frcrn_se_ovrl']
        demucs_ovrl = row['demucs_ovrl']
        
        # 找到最高分数和对应的数据集
        scores = {
            'raw': raw_ovrl,
            'mossformer': mossformer_ovrl,
            'frcrn_se': frcrn_se_ovrl,
            'demucs': demucs_ovrl
        }
        
        best_dataset = max(scores, key=scores.get)
        best_score = scores[best_dataset]
        
        # 确定源文件路径
        source_dataset_path = dataset_mapping[best_dataset]
        
        # 根据 category 确定子文件夹（保持首字母大写）
        # CSV中的category是 'Control' 或 'Dementia'
        category_folder = category  # 直接使用，因为CSV中已经是首字母大写
        
        # 构建源文件路径和目标文件路径
        source_file = os.path.join(source_dataset_path, category_folder, file_name)
        
        if category == 'Dementia':
            target_file = os.path.join(dementia_dir, file_name)
        else:  # Control
            target_file = os.path.join(control_dir, file_name)
        
        # 复制文件
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            stats[best_dataset] += 1
            stats['total'] += 1
            
            # 记录选择详情
            selection_details.append({
                'file_name': file_name,
                'category': category,
                'selected_dataset': best_dataset,
                'ovrl_score': best_score,
                'raw_ovrl': raw_ovrl,
                'mossformer_ovrl': mossformer_ovrl,
                'frcrn_se_ovrl': frcrn_se_ovrl,
                'demucs_ovrl': demucs_ovrl
            })
            
            if (idx + 1) % 50 == 0:
                print(f"  已处理 {idx + 1}/{len(df)} 个文件...")
        else:
            print(f"  警告: 找不到源文件: {source_file}")
            stats['missing'] += 1
    
    # 打印统计信息
    print(f"总共处理文件数: {stats['total']}")
    print(f"从 raw 选择: {stats['raw']} ({stats['raw']/stats['total']*100:.1f}%)")
    print(f"从 mossformer 选择: {stats['mossformer']} ({stats['mossformer']/stats['total']*100:.1f}%)")
    print(f"从 frcrn_se 选择: {stats['frcrn_se']} ({stats['frcrn_se']/stats['total']*100:.1f}%)")
    print(f"从 demucs 选择: {stats['demucs']} ({stats['demucs']/stats['total']*100:.1f}%)")
    print(f"缺失文件数: {stats['missing']}")
    
    # 创建 CSV 统计报告文件
    report_csv_path = os.path.join(output_dir, "selection_report.csv")
    with open(report_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'file_name', 'category', 'selected_dataset', 'ovrl_score',
            'raw_ovrl', 'mossformer_ovrl', 'frcrn_se_ovrl', 'demucs_ovrl'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selection_details)


def main():
    
    csv_path = "audio_quality_evaluation.csv"
    output_dir = "../ad_detection/data/denoised/Pitt-Best"
    
    print("Pitt-best Construction")
    print("="*60)
    print(f"CSV文件路径: {csv_path}")
    print(f"输出目录: {output_dir}")
    
    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        print(f"\n错误: 找不到CSV文件: {csv_path}")
        print("请检查路径是否正确")
        return
    
    # 检查数据源目录是否存在
    data_dirs = {
        'raw': '../ad_detection/data/raw/Pitt',
        'mossformer': '../ad_detection/data/denoised/Pitt-mossformer',
        'frcrn_se': '../ad_detection/data/denoised/Pitt-FRCRN_SE',
        'demucs': '../ad_detection/data/denoised/Pitt-Demucs'
    }
    
    for name, path in data_dirs.items():
        if not os.path.exists(path):
            print(f"\n警告: 找不到 {name} 数据目录: {path}")
    
    print()
    
    # 创建数据集
    create_pitt_best_dataset(csv_path, output_dir)


if __name__ == "__main__":
    main()
