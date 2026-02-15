"""
音频质量评估工具
功能：使用 DNSMOS 评估原始音频、MossFormer、FRCRN_SE 和 Demucs 降噪后的音频质量
生成横向对比的 CSV 表格
"""

import os
import csv
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
import onnxruntime as ort
import requests
from typing import Dict, Tuple, Optional


class DNSMOSEvaluator:
    """DNSMOS P.835 音频质量评估器"""
    
    MODEL_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
    MODEL_PATH = "dnsmos_model.onnx"
    TARGET_SR = 16000  # DNSMOS 模型要求的采样率
    INPUT_LENGTH = 9.01  # 模型要求的音频长度（秒）
    
    def __init__(self):
        """初始化评估器，下载并加载 ONNX 模型"""
        self._download_model()
        self.session = ort.InferenceSession(self.MODEL_PATH)
        self.input_name = self.session.get_inputs()[0].name
    
    def _download_model(self):
        if os.path.exists(self.MODEL_PATH):
            print(f"DNSMOS 模型已存在: {self.MODEL_PATH}")
            return
        
        print(f"正在下载 DNSMOS 模型...")
        try:
            response = requests.get(self.MODEL_URL, timeout=30)
            response.raise_for_status()
            with open(self.MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print(f"DNSMOS 模型下载成功: {self.MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(f"下载 DNSMOS 模型失败: {e}")
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        加载并预处理音频文件
        
        参数:
            audio_path (str): 音频文件路径
            
        返回:
            np.ndarray: 预处理后的音频数据 (单声道, 16kHz)
        """
        # 读取音频文件
        audio, sr = sf.read(audio_path)
        
        # 如果是立体声，转换为单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 重采样到 16kHz（如果需要）
        if sr != self.TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.TARGET_SR)
        
        # 归一化到 [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        # 确保在合理范围内
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _process_audio_segment(self, audio: np.ndarray) -> Dict[str, float]:
        """
        处理单个音频片段（固定长度）
        
        参数:
            audio (np.ndarray): 音频数据
            
        返回:
            Dict[str, float]: 包含 'ovrl', 'sig', 'bak' 三个分数的字典
        """
        desired_length = int(self.INPUT_LENGTH * self.TARGET_SR)
        
        # 如果音频太短，进行填充
        if len(audio) < desired_length:
            audio = np.pad(audio, (0, desired_length - len(audio)), mode='constant')
        # 如果音频太长，截取前面部分
        elif len(audio) > desired_length:
            audio = audio[:desired_length]
        
        # 准备输入数据
        audio_input = audio.astype(np.float32).reshape(1, -1)
        
        # 运行推理
        outputs = self.session.run(None, {self.input_name: audio_input})
        
        # 提取结果
        sig_score = float(outputs[0][0][0])
        bak_score = float(outputs[0][0][1])
        ovrl_score = float(outputs[0][0][2])
        
        return {
            'ovrl': ovrl_score,
            'sig': sig_score,
            'bak': bak_score
        }
    
    def evaluate(self, audio_path: str) -> Dict[str, float]:
        """
        评估音频质量（支持任意长度的音频）
        对于长音频，将其分割成多个片段分别评估，然后取平均值
        
        参数:
            audio_path (str): 音频文件路径
            
        返回:
            Dict[str, float]: 包含 'ovrl', 'sig', 'bak' 三个分数的字典
        """
        try:
            # 加载并预处理音频
            audio = self.load_audio(audio_path)
            
            desired_length = int(self.INPUT_LENGTH * self.TARGET_SR)
            
            # 如果音频长度小于等于所需长度，直接处理
            if len(audio) <= desired_length:
                return self._process_audio_segment(audio)
            
            # 对于长音频，分割成多个片段
            num_segments = int(np.ceil(len(audio) / desired_length))
            all_scores = {'ovrl': [], 'sig': [], 'bak': []}
            
            for i in range(num_segments):
                start_idx = i * desired_length
                end_idx = min((i + 1) * desired_length, len(audio))
                segment = audio[start_idx:end_idx]
                
                # 评估片段
                scores = self._process_audio_segment(segment)
                all_scores['ovrl'].append(scores['ovrl'])
                all_scores['sig'].append(scores['sig'])
                all_scores['bak'].append(scores['bak'])
            
            # 返回平均分数
            return {
                'ovrl': np.mean(all_scores['ovrl']),
                'sig': np.mean(all_scores['sig']),
                'bak': np.mean(all_scores['bak'])
            }
        except Exception as e:
            print(f"评估 {audio_path} 时出错: {e}")
            return None


def process_audio_file(
    evaluator: DNSMOSEvaluator,
    file_name: str,
    category: str,
    raw_path: str,
    mossformer_path: str,
    frcrn_se_path: str,
    demucs_path: str
) -> Optional[Dict]:
    """
    处理单个音频文件的四个版本并评估质量
    
    参数:
        evaluator: DNSMOS 评估器实例
        file_name: 文件名
        category: 分类 (Control 或 Dementia)
        raw_path: 原始音频路径
        mossformer_path: MossFormer 降噪后音频路径
        frcrn_se_path: FRCRN_SE 降噪后音频路径
        demucs_path: Demucs 降噪后音频路径
        
    返回:
        Dict: 包含所有分数的字典，如果任何文件不存在或评估失败则返回 None
    """
    # 检查文件是否存在
    if not os.path.exists(raw_path):
        print(f"警告: 原始文件不存在 - {raw_path}")
        return None
    
    if not os.path.exists(mossformer_path):
        print(f"警告: MossFormer 文件不存在 - {mossformer_path}")
        return None
    
    if not os.path.exists(frcrn_se_path):
        print(f"警告: FRCRN_SE 文件不存在 - {frcrn_se_path}")
        return None
    
    if not os.path.exists(demucs_path):
        print(f"警告: Demucs 文件不存在 - {demucs_path}")
        return None
    
    # 评估四个版本
    raw_scores = evaluator.evaluate(raw_path)
    mossformer_scores = evaluator.evaluate(mossformer_path)
    frcrn_se_scores = evaluator.evaluate(frcrn_se_path)
    demucs_scores = evaluator.evaluate(demucs_path)
    
    # 检查是否有评估失败
    if raw_scores is None or mossformer_scores is None or frcrn_se_scores is None or demucs_scores is None:
        return None
    
    # 返回所有分数（不计算gain）
    result = {
        'file_name': file_name,
        'category': category,
        'raw_ovrl': raw_scores['ovrl'],
        'raw_sig': raw_scores['sig'],
        'raw_bak': raw_scores['bak'],
        'mossformer_ovrl': mossformer_scores['ovrl'],
        'mossformer_sig': mossformer_scores['sig'],
        'mossformer_bak': mossformer_scores['bak'],
        'frcrn_se_ovrl': frcrn_se_scores['ovrl'],
        'frcrn_se_sig': frcrn_se_scores['sig'],
        'frcrn_se_bak': frcrn_se_scores['bak'],
        'demucs_ovrl': demucs_scores['ovrl'],
        'demucs_sig': demucs_scores['sig'],
        'demucs_bak': demucs_scores['bak'],
    }
    
    return result


def process_dataset(
    raw_dir: str,
    mossformer_dir: str,
    frcrn_se_dir: str,
    demucs_dir: str,
    output_csv: str
):
    """
    批量处理整个数据集，生成 CSV 报告
    
    参数:
        raw_dir: 原始音频根目录
        mossformer_dir: MossFormer 降噪后音频根目录
        frcrn_se_dir: FRCRN_SE 降噪后音频根目录
        demucs_dir: Demucs 降噪后音频根目录
        output_csv: 输出 CSV 文件路径
    """
    raw_dir = Path(raw_dir)
    mossformer_dir = Path(mossformer_dir)
    frcrn_se_dir = Path(frcrn_se_dir)
    demucs_dir = Path(demucs_dir)
    
    # 初始化评估器
    evaluator = DNSMOSEvaluator()
    
    # 子文件夹
    subdirs = ['Control', 'Dementia']
    
    # 收集所有结果
    all_results = []
    
    # 遍历每个子文件夹
    for subdir in subdirs:
        raw_subdir = raw_dir / subdir
        mossformer_subdir = mossformer_dir / subdir
        frcrn_se_subdir = frcrn_se_dir / subdir
        demucs_subdir = demucs_dir / subdir
        
        # 获取所有原始音频文件
        raw_files = sorted(list(raw_subdir.glob('*.wav')))
        
        print(f"\n处理 {subdir} 类别，共 {len(raw_files)} 个文件")
        
        # 使用进度条显示处理进度
        for raw_file in tqdm(raw_files, desc=f"评估 {subdir}"):
            file_name = raw_file.name
            
            # 构建对应的文件路径
            mossformer_file = mossformer_subdir / file_name
            frcrn_se_file = frcrn_se_subdir / file_name
            demucs_file = demucs_subdir / file_name
            
            # 处理音频文件
            result = process_audio_file(
                evaluator,
                file_name,
                subdir,
                str(raw_file),
                str(mossformer_file),
                str(frcrn_se_file),
                str(demucs_file)
            )
            
            if result is not None:
                all_results.append(result)
    
    # 写入 CSV 文件
    print(f"\n写入结果到 {output_csv}...")
    fieldnames = [
        'file_name', 'category',
        'raw_ovrl', 'raw_sig', 'raw_bak',
        'mossformer_ovrl', 'mossformer_sig', 'mossformer_bak',
        'frcrn_se_ovrl', 'frcrn_se_sig', 'frcrn_se_bak',
        'demucs_ovrl', 'demucs_sig', 'demucs_bak'
    ]
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

if __name__ == "__main__":
    # 设置路径
    raw_directory = "../ad_detection/data/raw/Pitt"
    mossformer_directory = "../ad_detection/data/denoised/Pitt-MossFormer"
    frcrn_se_directory = "../ad_detection/data/denoised/Pitt-FRCRN_SE"
    demucs_directory = "../ad_detection/data/denoised/Pitt-Demucs"
    output_csv_file = "audio_quality_evaluation.csv"
    
    # 执行处理
    process_dataset(
        raw_directory,
        mossformer_directory,
        frcrn_se_directory,
        demucs_directory,
        output_csv_file
    )
