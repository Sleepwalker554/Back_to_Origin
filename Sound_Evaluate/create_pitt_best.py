import pandas as pd
import os
import shutil
import csv
from pathlib import Path


def create_pitt_best_dataset(
    csv_path: str
):
    """
    Parameters:
        csv_path: audio_quality_evaluation.csv Path
        output_dir: defult ad_detection/data/denoised/Pitt-Best
    """
    
    df = pd.read_csv(csv_path)
    
    output_dir = "../ad_detection/data/denoised/Pitt-Best"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Subfolder： Dementia and Control
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
    
    selection_details = []
    
    for idx, row in df.iterrows():
        file_name = row['file_name']
        category = row['category']  # 'Control' or 'Dementia'
        
        # Ovrl score for each dataset
        mossformer_ovrl = row['mossformer_ovrl']
        frcrn_se_ovrl = row['frcrn_se_ovrl']
        demucs_ovrl = row['demucs_ovrl']
        raw_ovrl = row['raw_ovrl']
        
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
        else:
            print(f"  Warning Cannot find source file: {source_file}")
            stats['missing'] += 1
    
    print(f"Totally Process: {stats['total']}")
    print(f"From raw: {stats['raw']} ({stats['raw']/stats['total']*100:.1f}%)")
    print(f"From MossFormer: {stats['mossformer']} ({stats['mossformer']/stats['total']*100:.1f}%)")
    print(f"From frcrn_se: {stats['frcrn_se']} ({stats['frcrn_se']/stats['total']*100:.1f}%)")
    print(f"From demucs: {stats['demucs']} ({stats['demucs']/stats['total']*100:.1f}%)")
    print(f"Missing files: {stats['missing']}")
    
    # Constract CSV file
    report_csv_path = os.path.join(output_dir, "selection_report.csv")
    with open(report_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'file_name', 'category', 'selected_dataset', 'ovrl_score',
            'raw_ovrl', 'mossformer_ovrl', 'frcrn_se_ovrl', 'demucs_ovrl'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selection_details)