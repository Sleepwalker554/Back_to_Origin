"""
音频相减工具
功能：从 raw/Pitt 音频文件中减去 Pitt-mossformer 音频文件，生成残差音频
"""

import os
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from scipy import signal


def subtract_audio_files(raw_path, denoised_path, output_path):
    """
    从原始音频中减去降噪后的音频，得到噪声/残差部分
    
    参数:
        raw_path (str): 原始音频文件路径
        denoised_path (str): 降噪后音频文件路径
        output_path (str): 输出音频文件路径
    """
    # 读取两个音频文件
    raw_audio, raw_sr = sf.read(raw_path)
    denoised_audio, denoised_sr = sf.read(denoised_path)
    
    # 如果采样率不同，将原始音频重采样到降噪后的采样率
    if raw_sr != denoised_sr:
        num_samples = int(len(raw_audio) * denoised_sr / raw_sr)
        raw_audio = signal.resample(raw_audio, num_samples)
        raw_sr = denoised_sr
    
    # 对齐长度（处理可能的微小差异）
    min_len = min(len(raw_audio), len(denoised_audio))
    raw_audio = raw_audio[:min_len]
    denoised_audio = denoised_audio[:min_len]
    
    # 音频相减：原始音频 - 降噪后的音频 = 噪声部分
    residual_audio = raw_audio - denoised_audio
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存结果音频
    sf.write(output_path, residual_audio, raw_sr)


def process_dataset(raw_dir, denoised_dir, output_dir):
    """
    批量处理整个数据集，保持文件夹结构
    
    参数:
        raw_dir (str): 原始音频根目录
        denoised_dir (str): 降噪后音频根目录
        output_dir (str): 输出目录
    """
    raw_dir = Path(raw_dir)
    denoised_dir = Path(denoised_dir)
    output_dir = Path(output_dir)
    
    # 需要处理的子文件夹
    subdirs = ['Control', 'Dementia']
    
    # 遍历每个子文件夹（Control 和 Dementia）
    for subdir in subdirs:
        raw_subdir = raw_dir / subdir
        denoised_subdir = denoised_dir / subdir
        output_subdir = output_dir / subdir
        
        # 获取所有音频文件
        raw_files = list(raw_subdir.glob('*.wav'))
        
        # 使用进度条显示处理进度
        for raw_file in tqdm(raw_files, desc=f"处理 {subdir}"):
            # 构建对应的降噪文件路径和输出文件路径
            denoised_file = denoised_subdir / raw_file.name
            output_file = output_subdir / raw_file.name
            
            # 检查降噪文件是否存在
            if not denoised_file.exists():
                continue
            
            # 执行音频相减
            subtract_audio_files(str(raw_file), str(denoised_file), str(output_file))


if __name__ == "__main__":
    # 设置路径
    raw_directory = "ad_detection/data/raw/Pitt"
    denoised_directory = "ad_detection/data/denoised/Pitt-mossformer"
    output_directory = "ad_detection/data/residual/Pitt-residual-mossformer"
    
    # 执行处理
    # process_dataset(raw_directory, denoised_directory, output_directory)

    raw_directory1 = "ad_detection/data/raw/Pitt"
    denoised_directory1 = "ad_detection/data/denoised/Pitt-FRCRN_SE"
    output_directory1 = "ad_detection/data/residual/Pitt-residual-FRCRN_SE"

    # 执行处理
    process_dataset(raw_directory1, denoised_directory1, output_directory1)
