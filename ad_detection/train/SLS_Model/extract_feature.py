import csv
import torch
import librosa
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from typing import Union, Optional
from XLSR_model.model import SSLModel
from .config import SECOND_LENGTH
from utils.config import SAMPLING_RATE


def _stack_layers(layerresult) -> torch.Tensor:
    """
    layerresult: list of length L, each entry (layer_hidden, attn) where
                 layer_hidden has shape (T, B=1, 1024).
    Returns: tensor of shape (L, T, 1024).
    """
    feats = [l[0].squeeze(1) for l in layerresult]  # each (T, 1024)
    return torch.stack(feats, dim=0)


def extract_features_from_csv(
    csv_path: Union[str, Path],
    split_name: str,
    raw_audio_dir: Union[str, Path],
    sls_features_dir: Union[str, Path],
    device: str = "cpu",
    ssl_model: Optional[SSLModel] = None,
    freeze_xlsr: bool = True,
):
    """
    Extract all-layer SLS features for a CSV split and save.
    """
    csv_path = Path(csv_path)
    raw_audio_dir = Path(raw_audio_dir)
    sls_features_dir = Path(sls_features_dir)

    sls_features_dir.mkdir(parents=True, exist_ok=True)

    if ssl_model is None:
        ssl_model = SSLModel(device, freeze_xlsr=freeze_xlsr)

    print(f"\n=== Extracting SLS features for {split_name} ===")

    extracted = 0
    skipped = 0
    errors = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in tqdm(rows, desc=f"Extracting {split_name}"):
        session_id = row['session_id']
        ad = int(row['ad'])

        folder = "Control" if ad == 0 else "Dementia"
        audio_path_wav = raw_audio_dir / folder / f"{session_id}.wav"
        audio_path_mp3 = raw_audio_dir / folder / f"{session_id}.mp3"

        if audio_path_wav.exists():
            audio_path = audio_path_wav
        elif audio_path_mp3.exists():
            audio_path = audio_path_mp3
        else:
            print(f"\nError: Audio file does not exist: {session_id}")
            errors += 1
            continue

        sls_path = sls_features_dir / f"{session_id}.sls.pt"

        if sls_path.exists():
            skipped += 1
            continue

        try:
            audio_np, _ = librosa.load(
                str(audio_path),
                sr=SAMPLING_RATE,
                res_type="kaiser_best"
            )
            audio_np = librosa.to_mono(audio_np)
            audio_np = np.float32(audio_np)

            max_length = SAMPLING_RATE * SECOND_LENGTH
            if len(audio_np) > max_length:
                audio_np = audio_np[:max_length]

            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(device)

            _, layerresult = ssl_model.extract_feat(audio_tensor)

            feat = _stack_layers(layerresult)  # (L, T, 1024)
            assert feat.shape[0] in (24, 25), (
                f"Unexpected layer count {feat.shape[0]} for {session_id} "
                f"(expected 24 or 25)"
            )

            feat = feat.cpu().detach().half()  # fp16 to halve disk usage
            torch.save(feat, sls_path)
            extracted += 1

        except Exception as e:
            print(f"\nError: Extraction failed for {session_id}: {e}")
            errors += 1
            continue

    print(f"Done: {extracted} extracted, {skipped} skipped, {errors} errors (total {len(rows)})")

    return extracted, skipped, len(rows)


def extract_feature(
    train_csv: Union[str, Path],
    val_csv: Union[str, Path],
    raw_audio_dir: Union[str, Path],
    sls_features_dir: Union[str, Path],
    device: str = "cpu",
    freeze_xlsr: bool = True,
):
    """Create SSLModel and extract SLS features for train/val splits."""
    ssl_model = SSLModel(device, freeze_xlsr=freeze_xlsr)

    extract_features_from_csv(
        csv_path=train_csv,
        split_name="Train Set",
        raw_audio_dir=raw_audio_dir,
        sls_features_dir=sls_features_dir,
        device=device,
        ssl_model=ssl_model,
    )

    extract_features_from_csv(
        csv_path=val_csv,
        split_name="Val Set",
        raw_audio_dir=raw_audio_dir,
        sls_features_dir=sls_features_dir,
        device=device,
        ssl_model=ssl_model,
    )

    return ssl_model
