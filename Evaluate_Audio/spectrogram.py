import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def show_difference_spectrograms(audio_id, category='Control', spec_type='mel'):
    """
    audio_id: 音频文件名，例如 '002-0.wav'
    category: 'Control' 或 'Dementia'
    spec_type: 'mel' 或 'stft'
    """
    raw_path = Path('../ad_detection/data/raw/Pitt') / category / audio_id
    
    denoised_datasets = {
        'Diff: Raw - Demucs': Path('../ad_detection/data/denoised/Pitt-Demucs') / category / audio_id,
        'Diff: Raw - FRCRN_SE': Path('../ad_detection/data/denoised/Pitt-FRCRN_SE') / category / audio_id,
        'Diff: Raw - MossFormer': Path('../ad_detection/data/denoised/Pitt-MossFormer') / category / audio_id,
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    y_raw, sr_raw = librosa.load(raw_path, sr=None)
    
    for idx, (name, denoised_path) in enumerate(denoised_datasets.items()):
        y_denoised, sr_denoised = librosa.load(denoised_path, sr=None)
        
        if sr_raw != sr_denoised:
            y_denoised = librosa.resample(y_denoised, orig_sr=sr_denoised, target_sr=sr_raw)
        
        min_len = min(len(y_raw), len(y_denoised))
        y_raw_trimmed = y_raw[:min_len]
        y_denoised_trimmed = y_denoised[:min_len]
        
        y_diff = y_raw_trimmed - y_denoised_trimmed
        if spec_type == 'mel':
            spec = librosa.feature.melspectrogram(y=y_diff, sr=sr_raw)
            spec_db = librosa.power_to_db(spec, ref=np.max)
            librosa.display.specshow(spec_db, sr=sr_raw, x_axis='time', y_axis='mel', ax=axes[idx])
            axes[idx].set_ylabel('Frequency (Hz)')
        else:  # stft
            D = librosa.stft(y_diff)
            D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            librosa.display.specshow(D_db, sr=sr_raw, x_axis='time', y_axis='hz', ax=axes[idx])
            axes[idx].set_ylabel('Frequency (Hz)')
        
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Time (s)')
    
    spec_name = 'Mel' if spec_type == 'mel' else 'STFT'
    plt.suptitle(f'Difference {spec_name} Spectrograms: {audio_id} ({category})', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def show_mel_spectrograms(audio_id, category='Control', base_path='../ad_detection/data'):
    """
    audio_id: 音频文件名，例如 '002-0.wav'
    category: 'Control' 或 'Dementia'
    """
    datasets = {
        'Pitt': Path('../ad_detection/data/raw/Pitt') / category / audio_id,
        'Pitt-Demucs': Path('../ad_detection/data/denoised/Pitt-Demucs') / category / audio_id,
        'Pitt-FRCRN_SE': Path('../ad_detection/data/denoised/Pitt-FRCRN_SE') / category / audio_id,
        'Pitt-MossFormer': Path('../ad_detection/data/denoised/Pitt-MossFormer') / category / audio_id,
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, path) in enumerate(datasets.items()):
        if path.exists():
            y, sr = librosa.load(path, sr=None)
            
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[idx])
            axes[idx].set_title(name, fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Time (s)')
            axes[idx].set_ylabel('Frequency (Hz)')
        else:
            axes[idx].text(0.5, 0.5, f'File not found:\\n{path.name}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(name, fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Mel Spectrograms: {audio_id} ({category})', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def show_stft_spectrograms(audio_id, category='Control'):
    """
    audio_id: 音频文件名，例如 '002-0.wav'
    category: 'Control' 或 'Dementia'
    """
    # 定义四个数据集路径
    datasets = {
        'Pitt': Path('../ad_detection/data/raw/Pitt') / category / audio_id,
        'Pitt-Demucs': Path('../ad_detection/data/denoised/Pitt-Demucs') / category / audio_id,
        'Pitt-FRCRN_SE': Path('../ad_detection/data/denoised/Pitt-FRCRN_SE') / category / audio_id,
        'Pitt-MossFormer': Path('../ad_detection/data/denoised/Pitt-MossFormer') / category / audio_id,
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, path) in enumerate(datasets.items()):
        # 加载音频
        y, sr = librosa.load(path, sr=None)
        
        # 计算 STFT
        D = librosa.stft(y)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # 绘制
        librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[idx])
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Time (s)')
        axes[idx].set_ylabel('Frequency (Hz)')
    
    plt.suptitle(f'STFT Spectrograms: {audio_id} ({category})', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def show_log_mel_spectrograms(audio_id, category='Control'):
    """
    audio_id: 音频文件名，例如 '002-0.wav'
    category: 'Control' 或 'Dementia'
    """
    datasets = {
        'Pitt': Path('../ad_detection/data/raw/Pitt') / category / audio_id,
        'Pitt-Demucs': Path('../ad_detection/data/denoised/Pitt-Demucs') / category / audio_id,
        'Pitt-FRCRN_SE': Path('../ad_detection/data/denoised/Pitt-FRCRN_SE') / category / audio_id,
        'Pitt-MossFormer': Path('../ad_detection/data/denoised/Pitt-MossFormer') / category / audio_id,
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, path) in enumerate(datasets.items()):
        if path.exists():
            y, sr = librosa.load(path, sr=None)
            
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', ax=axes[idx], cmap='viridis')
            axes[idx].set_title(name, fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Time (s)')
            axes[idx].set_ylabel('Frequency (Hz)')
        else:
            axes[idx].text(0.5, 0.5, f'File not found:\\n{path.name}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(name, fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Log-Mel Spectrograms: {audio_id} ({category})', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def show_difference_log_mel_spectrograms(audio_id, category='Control'):
    """
    audio_id: 音频文件名，例如 '002-0.wav'
    category: 'Control' 或 'Dementia'
    """
    raw_path = Path('../ad_detection/data/raw/Pitt') / category / audio_id
    
    denoised_datasets = {
        'Diff: Raw - Demucs': Path('../ad_detection/data/denoised/Pitt-Demucs') / category / audio_id,
        'Diff: Raw - FRCRN_SE': Path('../ad_detection/data/denoised/Pitt-FRCRN_SE') / category / audio_id,
        'Diff: Raw - MossFormer': Path('../ad_detection/data/denoised/Pitt-MossFormer') / category / audio_id,
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    y_raw, sr_raw = librosa.load(raw_path, sr=None)
    
    for idx, (name, denoised_path) in enumerate(denoised_datasets.items()):
        y_denoised, sr_denoised = librosa.load(denoised_path, sr=None)
        
        if sr_raw != sr_denoised:
            y_denoised = librosa.resample(y_denoised, orig_sr=sr_denoised, target_sr=sr_raw)
        
        min_len = min(len(y_raw), len(y_denoised))
        y_raw_trimmed = y_raw[:min_len]
        y_denoised_trimmed = y_denoised[:min_len]
        
        y_diff = y_raw_trimmed - y_denoised_trimmed
        
        mel_spec = librosa.feature.melspectrogram(y=y_diff, sr=sr_raw)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        librosa.display.specshow(log_mel_spec, sr=sr_raw, x_axis='time', y_axis='mel', ax=axes[idx], cmap='viridis')
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Time (s)')
        axes[idx].set_ylabel('Frequency (Hz)')
    
    plt.suptitle(f'Difference Log-Mel Spectrograms: {audio_id} ({category})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def show_waveform_comparison(audio_id, category='Control'):
    """
    audio_id: 音频文件名，例如 '002-0.wav'
    category: 'Control' 或 'Dementia'
    """
    raw_path = Path('../ad_detection/data/raw/Pitt') / category / audio_id
    
    denoised_datasets = {
        'Raw vs Demucs': Path('../ad_detection/data/denoised/Pitt-Demucs') / category / audio_id,
        'Raw vs FRCRN_SE': Path('../ad_detection/data/denoised/Pitt-FRCRN_SE') / category / audio_id,
        'Raw vs MossFormer': Path('../ad_detection/data/denoised/Pitt-MossFormer') / category / audio_id,
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    y_raw, sr_raw = librosa.load(raw_path, sr=None)
    time_raw = np.arange(len(y_raw)) / sr_raw
    
    for idx, (name, denoised_path) in enumerate(denoised_datasets.items()):
        y_denoised, sr_denoised = librosa.load(denoised_path, sr=None)
        
        if sr_raw != sr_denoised:
            y_denoised = librosa.resample(y_denoised, orig_sr=sr_denoised, target_sr=sr_raw)
        
        min_len = min(len(y_raw), len(y_denoised))
        y_raw_trimmed = y_raw[:min_len]
        y_denoised_trimmed = y_denoised[:min_len]
        time_trimmed = time_raw[:min_len]
        
        axes[idx].plot(time_trimmed, y_raw_trimmed, color='blue', alpha=0.6, linewidth=0.5, label='Raw')
        axes[idx].plot(time_trimmed, y_denoised_trimmed, color='red', alpha=0.6, linewidth=0.5, label='Denoised')
        
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Time (s)')
        axes[idx].set_ylabel('Amplitude')
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f'Waveform Comparison: {audio_id} ({category})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()