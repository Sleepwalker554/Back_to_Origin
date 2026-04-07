import os
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import onnxruntime as ort
import requests
from typing import Dict, Optional, List

SUPPORTED_EXTENSIONS = {'.wav', '.mp3'}


class DNSMOSEvaluator:
    MODEL_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
    MODEL_PATH = "dnsmos_model.onnx"
    TARGET_SR = 16000
    INPUT_LENGTH = 9.01

    def __init__(self):
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
        ext = Path(audio_path).suffix.lower()
        if ext == '.mp3':
            audio, sr = librosa.load(audio_path, sr=self.TARGET_SR, mono=True)
            return np.clip(audio.astype(np.float32), -1.0, 1.0)

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


def _collect_audio_files(directory: Path) -> List[Path]:
    """递归收集目录下所有支持格式的音频文件"""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(files)


def evaluate_directory(audio_dir: str) -> pd.DataFrame:
    """
    评估单个目录下所有音频文件的 DNSMOS 分数。
    支持 wav 和 mp3 格式，会递归搜索子目录。
    """
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        raise FileNotFoundError(f"目录不存在: {audio_dir}")

    audio_files = _collect_audio_files(audio_path)
    if not audio_files:
        raise FileNotFoundError(
            f"目录中没有找到支持的音频文件 ({', '.join(SUPPORTED_EXTENSIONS)}): {audio_dir}"
        )

    print(f"目录: {audio_dir}")
    print(f"找到 {len(audio_files)} 个音频文件\n")

    evaluator = DNSMOSEvaluator()
    results = []

    for fpath in tqdm(audio_files, desc="评估中"):
        scores = evaluator.evaluate(str(fpath))
        if scores is not None:
            rel = fpath.relative_to(audio_path)
            results.append({
                'file': str(rel),
                'OVRL': round(scores['ovrl'], 4),
                'SIG': round(scores['sig'], 4),
                'BAK': round(scores['bak'], 4),
            })

    df = pd.DataFrame(results)

    if df.empty:
        print("没有成功评估的文件")
        return df

    avg = {
        'OVRL': round(df['OVRL'].mean(), 4),
        'SIG': round(df['SIG'].mean(), 4),
        'BAK': round(df['BAK'].mean(), 4),
    }

    print(f"\n=== DNSMOS 平均分 ({audio_dir}) ===")
    print(f"  OVRL (Overall):    {avg['OVRL']}")
    print(f"  SIG  (Signal):     {avg['SIG']}")
    print(f"  BAK  (Background): {avg['BAK']}")
    print(f"  共评估 {len(df)} 个文件")

    return avg
