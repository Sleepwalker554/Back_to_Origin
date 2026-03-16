import os
import csv
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
import onnxruntime as ort
import requests
from typing import Dict, Optional


class DNSMOSEvaluator:
    MODEL_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
    MODEL_PATH = "dnsmos_model.onnx"
    TARGET_SR = 16000
    INPUT_LENGTH = 9.01  # 模型要求的音频长度（秒）

    def __init__(self):
        """初始化评估器，下载并加载 ONNX 模型"""
        self._download_model()
        self.session = ort.InferenceSession(self.MODEL_PATH)
        self.input_name = self.session.get_inputs()[0].name

    def _download_model(self):
        if os.path.exists(self.MODEL_PATH):
            return

        print(f"Downloading DNSMOS...")
        try:
            response = requests.get(self.MODEL_URL, timeout=30)
            response.raise_for_status()
            with open(self.MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print(f"DNSMOS Downloaded and saved to: {self.MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(f"Failed to download DNSMOS: {e}")

    def load_audio(self, audio_path: str) -> np.ndarray:
        audio, sr = sf.read(audio_path)

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        if sr != self.TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.TARGET_SR)

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        audio = np.clip(audio, -1.0, 1.0)
        return audio

    def _process_audio_segment(self, audio: np.ndarray) -> Dict[str, float]:
        desired_length = int(self.INPUT_LENGTH * self.TARGET_SR)

        if len(audio) < desired_length:
            audio = np.pad(audio, (0, desired_length - len(audio)), mode='constant')
        elif len(audio) > desired_length:
            audio = audio[:desired_length]

        audio_input = audio.astype(np.float32).reshape(1, -1)
        outputs = self.session.run(None, {self.input_name: audio_input})

        sig_score = float(outputs[0][0][0])
        bak_score = float(outputs[0][0][1])
        ovrl_score = float(outputs[0][0][2])

        return {'ovrl': ovrl_score, 'sig': sig_score, 'bak': bak_score}

    def evaluate(self, audio_path: str) -> Optional[Dict[str, float]]:
        try:
            audio = self.load_audio(audio_path)
            desired_length = int(self.INPUT_LENGTH * self.TARGET_SR)

            if len(audio) <= desired_length:
                return self._process_audio_segment(audio)

            num_segments = int(np.ceil(len(audio) / desired_length))
            all_scores = {'ovrl': [], 'sig': [], 'bak': []}

            for i in range(num_segments):
                start_idx = i * desired_length
                end_idx = min((i + 1) * desired_length, len(audio))
                segment = audio[start_idx:end_idx]

                scores = self._process_audio_segment(segment)
                all_scores['ovrl'].append(scores['ovrl'])
                all_scores['sig'].append(scores['sig'])
                all_scores['bak'].append(scores['bak'])

            return {
                'ovrl': np.mean(all_scores['ovrl']),
                'sig': np.mean(all_scores['sig']),
                'bak': np.mean(all_scores['bak'])
            }
        except Exception as e:
            print(f"评估 {audio_path} 时出错: {e}")
            return None


# 所有降噪方法名称
METHOD_NAMES = ['raw', 'mossformer', 'frcrn_se', 'demucs', 'denoiser', 'resemble']


def process_audio_file(
    evaluator: DNSMOSEvaluator,
    file_name: str,
    category: str,
    paths: Dict[str, str]
) -> Optional[Dict]:
    """
    处理单个音频文件的所有版本并评估质量
    """
    result = {'file_name': file_name, 'category': category}

    for method in METHOD_NAMES:
        scores = evaluator.evaluate(paths[method])
        if scores is None:
            return None
        result[f'{method}_ovrl'] = scores['ovrl']
        result[f'{method}_sig'] = scores['sig']
        result[f'{method}_bak'] = scores['bak']

    return result


def process_dataset(
    raw_dir: str,
    mossformer_dir: str,
    frcrn_se_dir: str,
    demucs_dir: str,
    denoiser_dir: str,
    resemble_dir: str
):
    """
    批量处理整个数据集，生成 CSV 报告
    """
    dirs = {
        'raw': Path(raw_dir),
        'mossformer': Path(mossformer_dir),
        'frcrn_se': Path(frcrn_se_dir),
        'demucs': Path(demucs_dir),
        'denoiser': Path(denoiser_dir),
        'resemble': Path(resemble_dir),
    }
    output_csv = Path("audio_quality_evaluation.csv")

    evaluator = DNSMOSEvaluator()
    subdirs = ['Control', 'Dementia']
    all_results = []

    for subdir in subdirs:
        raw_subdir = dirs['raw'] / subdir
        raw_files = sorted(list(raw_subdir.glob('*.wav')))
        print(f"\n处理 {subdir} 类别，共 {len(raw_files)} 个文件")

        for raw_file in tqdm(raw_files, desc=f"评估 {subdir}"):
            file_name = raw_file.name

            paths = {method: str(dirs[method] / subdir / file_name) for method in METHOD_NAMES}

            result = process_audio_file(evaluator, file_name, subdir, paths)
            if result is not None:
                all_results.append(result)

    # 写入 CSV 文件
    print(f"\n写入结果到 {output_csv}...")
    fieldnames = ['file_name', 'category']
    for method in METHOD_NAMES:
        fieldnames.extend([f'{method}_ovrl', f'{method}_sig', f'{method}_bak'])

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    return all_results
