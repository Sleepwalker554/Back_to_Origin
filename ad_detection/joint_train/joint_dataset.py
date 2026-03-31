import csv
import torch
import librosa
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from joint_config import (
    PROJECT_ROOT, BATCH_SIZE, NUM_WORKERS,
    SAMPLING_RATE, JOINT_SECOND_LENGTH
)


class JointTrainingDataset(Dataset):
    """
    加载 (raw音频, MossFormer pseudo clean音频, AD标签) 三元组
    """

    def __init__(self, csv_path: Path, raw_audio_dir: Path, clean_audio_dir: Path):
        """
        Args:
            csv_path: 现有Pitt CSV (含 session_id, ad 列)
            raw_audio_dir: 原始音频目录 (含 Control/Dementia 子文件夹)
            clean_audio_dir: MossFormer降噪音频目录 (含 Control/Dementia 子文件夹)
        """
        super().__init__()
        self.raw_audio_dir = Path(raw_audio_dir)
        self.clean_audio_dir = Path(clean_audio_dir)
        self.max_length = SAMPLING_RATE * JOINT_SECOND_LENGTH

        self.samples = []  # [(session_id, label), ...]
        self._load_csv(csv_path)

    def _load_csv(self, csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                session_id = row['session_id']
                label = int(row['ad'])
                self.samples.append((session_id, label))

        num_control = sum(1 for _, l in self.samples if l == 0)
        num_dementia = sum(1 for _, l in self.samples if l == 1)
        print(f"JointDataset: {len(self.samples)} samples "
              f"(Control: {num_control}, Dementia: {num_dementia})")

    def _load_audio(self, audio_dir, session_id, label):
        folder = "Control" if label == 0 else "Dementia"
        wav_path = audio_dir / folder / f"{session_id}.wav"
        mp3_path = audio_dir / folder / f"{session_id}.mp3"

        if wav_path.exists():
            audio_path = wav_path
        elif mp3_path.exists():
            audio_path = mp3_path
        else:
            raise FileNotFoundError(f"Audio not found: {wav_path}")

        audio, _ = librosa.load(str(audio_path), sr=SAMPLING_RATE, res_type="kaiser_best")
        audio = librosa.to_mono(audio)
        audio = np.float32(audio)

        # pad or truncate to max_length
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        elif len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')

        return audio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        session_id, label = self.samples[index]

        raw_audio = self._load_audio(self.raw_audio_dir, session_id, label)
        clean_audio = self._load_audio(self.clean_audio_dir, session_id, label)

        raw_tensor = torch.from_numpy(raw_audio)
        clean_tensor = torch.from_numpy(clean_audio)

        return raw_tensor, clean_tensor, label


def joint_collate_fn(batch):
    """
    Collate function: 所有音频已pad到相同长度，直接stack
    """
    raw_list, clean_list, label_list = zip(*batch)

    raw_batch = torch.stack(raw_list, dim=0)       # (B, T)
    clean_batch = torch.stack(clean_list, dim=0)   # (B, T)
    labels = torch.tensor(label_list, dtype=torch.long)  # (B,)

    return raw_batch, clean_batch, labels


def _make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=joint_collate_fn,
    )


def create_joint_dataloaders(
    train_csv: Path,
    val_csv: Path,
    raw_audio_dir: Path,
    clean_audio_dir: Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    train_dataset = JointTrainingDataset(train_csv, raw_audio_dir, clean_audio_dir)
    val_dataset = JointTrainingDataset(val_csv, raw_audio_dir, clean_audio_dir)

    train_loader = _make_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = _make_loader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
