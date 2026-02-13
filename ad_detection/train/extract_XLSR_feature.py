import csv
import torch
import librosa
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from typing import Union, Optional
from model import SSLModel
from config import SECOND_LENGTH, SAMPLING_RATE


def extract_features_from_csv(
    csv_path: Union[str, Path],
    split_name: str,
    raw_audio_dir: Union[str, Path],
    xlsr_features_dir: Union[str, Path],
    device: str = "cpu",
    ssl_model: Optional[SSLModel] = None,
    freeze_xlsr: bool = True,
):
    """
    Extract XLSR features for a CSV split.
    
    Args:
        csv_path: Path to the CSV file containing session_id and ad columns.
        split_name: Name of the split (for logging).
        raw_audio_dir: Directory containing Control/Dementia subfolders with audio files.
        xlsr_features_dir: Output directory for .xlsr.pt feature files.
        device: Device for inference (e.g., "cpu" or "cuda").
        ssl_model: Optional preloaded SSLModel to reuse across calls.
        freeze_xlsr: Whether to freeze XLSR parameters when creating a model.
    """
    csv_path = Path(csv_path)
    raw_audio_dir = Path(raw_audio_dir)
    xlsr_features_dir = Path(xlsr_features_dir)

    xlsr_features_dir.mkdir(parents=True, exist_ok=True)

    if ssl_model is None:
        print(f"Using device: {device}")
        ssl_model = SSLModel(device, freeze_xlsr=freeze_xlsr)

    print(f"\n============= Extracting XLSR features for {split_name} =============")

    extracted = 0
    skipped = 0
    errors = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"{len(rows)} Audio Files")

    for row in tqdm(rows, desc=f"Extracting {split_name}"):
        session_id = row['session_id']
        ad = int(row['ad'])

        # Build audio path 
        # from CSV's third column (ad) to determine audio path in Control or Dementia folder
        folder = "Control" if ad == 0 else "Dementia"
        audio_path_wav = raw_audio_dir / folder / f"{session_id}.wav"
        audio_path_mp3 = raw_audio_dir / folder / f"{session_id}.mp3"

        if audio_path_wav.exists():
            audio_path = audio_path_wav
        elif audio_path_mp3.exists():
            audio_path = audio_path_mp3
        else:
            audio_path = None
            print(f"\nError: Audio file does not exist: {session_id}")
            errors += 1
            continue

        # Build xlsr feature path
        xlsr_path = xlsr_features_dir / f"{session_id}.xlsr.pt"

        if xlsr_path.exists():
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

            # Make each Audio file same duration
            max_length = SAMPLING_RATE * SECOND_LENGTH
            if len(audio_np) > max_length:
                audio_np = audio_np[:max_length]
            # if len(audio_np) < max_length:
            #     audio_np = np.pad(audio_np, (0, max_length - len(audio_np)), mode='constant')
            
            # Convert audio to tensor
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(device)

            # Extract XLSR features
            # Extract XLSR embeddings
            emb, layerresult = ssl_model.extract_feat(audio_tensor)

            # Average pooling features (XLSR_FEATURE_DIM = 1024)
            # layery, fullfeature = XLSR_Average_Pooling(layerresult)

            # Save features
            # xlsr_features = layery[:, -1, :].cpu().detach()

            last_layer = layerresult[-1][0]  # (Time, Batch=1, Feature=1024)
            last_layer = last_layer.transpose(0, 1)  # (Batch=1, Time, Feature=1024)
            xlsr_features = last_layer.squeeze(0).cpu().detach()  # (Batch=1, Time, 1024) -> (Time, 1024)
            
            # Save features to file
            torch.save(xlsr_features, xlsr_path)
            extracted += 1

        except Exception as e:
            print(f"\nError: Extraction failed for {session_id}: {e}")
            errors += 1
            continue

    print(f"Successfully extracted: {extracted}")
    print(f"Already exists (skipped): {skipped}")
    print(f"Errors: {errors}")
    print(f"Total: {len(rows)}")

    return extracted, skipped, len(rows)
