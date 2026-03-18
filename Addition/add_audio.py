import os
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from scipy import signal


def add_audio_files(raw_path, denoised_path, output_path, raw_weight=1.0):
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

    # 音频相加：原始音频 * 系数 + 降噪后的音频
    residual_audio = raw_audio * raw_weight + denoised_audio
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存结果音频
    sf.write(output_path, residual_audio, raw_sr)


def process_dataset(raw_dir, denoised_dir, output_dir, raw_weight=1.0):
    raw_dir = Path(raw_dir)
    denoised_dir = Path(denoised_dir)
    output_dir = Path(output_dir)
    
    # 需要处理的子文件夹
    subdirs = ['Control', 'Dementia']
    
    skipped_files = []
    
    # 遍历每个子文件夹（Control 和 Dementia）
    for subdir in subdirs:
        raw_subdir = raw_dir / subdir
        denoised_subdir = denoised_dir / subdir
        output_subdir = output_dir / subdir
        
        # 获取所有音频文件
        raw_files = list(raw_subdir.glob('*.wav')) + list(raw_subdir.glob('*.mp3'))
        
        # 使用进度条显示处理进度
        for raw_file in tqdm(raw_files, desc=f"处理 {subdir}"):
            # 构建对应的降噪文件路径和输出文件路径
            denoised_file = denoised_subdir / (raw_file.stem + '.wav')
            output_file = output_subdir / raw_file.name
            
            # 检查降噪文件是否存在
            if not denoised_file.exists():
                skipped_files.append((raw_file.name, "降噪文件不存在"))
                continue
            
            # 检查文件是否为空
            if denoised_file.stat().st_size == 0:
                skipped_files.append((raw_file.name, "降噪文件为空"))
                continue
            
            if output_file.exists():
                continue

            try:
                add_audio_files(str(raw_file), str(denoised_file), str(output_file), raw_weight)
            except Exception as e:
                skipped_files.append((raw_file.name, f"处理失败: {str(e)}"))
                continue
    
    # 输出跳过文件的统计信息
    if skipped_files:
        print(f"\n跳过了 {len(skipped_files)} 个文件:")
        for filename, reason in skipped_files[:10]:
            print(f"  - {filename}: {reason}")
        if len(skipped_files) > 10:
            print(f"  ... 还有 {len(skipped_files) - 10} 个文件")
    
    return skipped_files