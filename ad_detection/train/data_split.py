import csv
from pathlib import Path
from random import Random
from typing import Tuple, Optional
from config import PROJECT_ROOT, RANDOM_SEED, TRAIN_SET_RATTIO

def create_train_val_split(
    raw_audio_dir: Path,
    dataset_name: str,
    feature_dir_name: str,
    train_set_ratio: float = TRAIN_SET_RATTIO,
    random_seed: int = RANDOM_SEED,
    xlsr: bool = False,
) -> Tuple[Path, Path]:
    """
    Create training and validation CSV files
    
    Args:
        raw_audio_dir: Directory containing raw audio files 
                       (with subfolders Control and Dementia)
        dataset_name: Dataset name (used to construct CSV paths)
        feature_dir_name: Feature directory name
        train_set_ratio: Ratio of training samples (default: 0.8)
        random_seed: Random seed (default: 42)
        xlsr: Whether to use XLSR feature mode 
    
    Returns:
        Tuple[Path, Path]: (train_csv_path, val_csv_path)
    """
    
    feature_dir_name = str(feature_dir_name)
    
    # Build CSV paths
    train_csv_path = PROJECT_ROOT / f"data/processed/{dataset_name}-xlsr-train.csv"
    val_csv_path = PROJECT_ROOT / f"data/processed/{dataset_name}-xlsr-val.csv"
    
    # Ensure output directories exist
    train_csv_path.parent.mkdir(parents=True, exist_ok=True)
    val_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check whether raw audio directory exists
    if not raw_audio_dir.exists():
        raise FileNotFoundError(f"Error: Raw audio directory does not exist: {raw_audio_dir}")
    
    # Collect all audio files
    control_samples = []
    dementia_samples = [] 

    # Process Control group
    control_dir = raw_audio_dir / "Control"
    if not control_dir.exists():
        raise FileNotFoundError(f"Error: Control dir does not exist: {control_dir}")
    
    # List all .wav and .mp3 files
    audio_files = list(control_dir.glob("*.wav")) + list(control_dir.glob("*.mp3"))
    for audio_file in sorted(audio_files):
        session_id = audio_file.stem
        control_samples.append({
            'session_id': session_id,
            'ad': 0
        })
    if len(control_samples) == 0:
        raise ValueError(f"Error: No control audio files found in {raw_audio_dir}") 
    
    # Process Dementia group
    dementia_dir = raw_audio_dir / "Dementia"
    if not dementia_dir.exists():
        raise FileNotFoundError(f"Error: Dementia dir does not exist: {dementia_dir}")
    
    # List all .wav and .mp3 files
    audio_files = list(dementia_dir.glob("*.wav")) + list(dementia_dir.glob("*.mp3"))
    for audio_file in sorted(audio_files):
        session_id = audio_file.stem
        dementia_samples.append({
            'session_id': session_id,
            'ad': 1
        })
    if len(dementia_samples) == 0:
        raise ValueError(f"Error: No control audio files found in {raw_audio_dir}")

    # Shuffle
    rdm = Random(random_seed) 
    rdm.shuffle(control_samples)
    rdm.shuffle(dementia_samples)
    
    # Split Control
    control_train_num = int(len(control_samples) * train_set_ratio)
    control_train = control_samples[:control_train_num]
    control_val = control_samples[control_train_num:]
    
    # Split Dementia
    dementia_train_num = int(len(dementia_samples) * train_set_ratio)
    dementia_train = dementia_samples[:dementia_train_num]
    dementia_val = dementia_samples[dementia_train_num:]
    
    # Combine
    train_samples = control_train + dementia_train
    val_samples = control_val + dementia_val
    
    # Shuffle again
    rdm.shuffle(train_samples)
    rdm.shuffle(val_samples)
    
    # Set column names and file extension based on feature type
    if xlsr:
        feature_col = 'xlsr_path'
        feature_ext = '.xlsr.pt'
    
    # Generate training CSV
    with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', feature_col, 'ad'])
        for sample in train_samples:
            session_id = sample['session_id']
            feature_path = f"{feature_dir_name}/{session_id}{feature_ext}"
            ad = sample['ad']
            writer.writerow([session_id, feature_path, ad])
    
    # Generate validation CSV
    with open(val_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', feature_col, 'ad'])
        for sample in val_samples:
            session_id = sample['session_id']
            feature_path = f"{feature_dir_name}/{session_id}{feature_ext}"
            ad = sample['ad']
            writer.writerow([session_id, feature_path, ad])
    
    # Print statistics
    print(f"============= {dataset_name} Train({TRAIN_SET_RATTIO*100}%) and Val({int((1-TRAIN_SET_RATTIO)*100 + 1)}%) Split Complete! =============")
    print(f"Training set: {len(train_samples)} samples (Control: {len(control_train)}, Dementia: {len(dementia_train)})")
    print(f"Validation set: {len(val_samples)} samples (Control: {len(control_val)}, Dementia: {len(dementia_val)})")

    return train_csv_path, val_csv_path


def create_split(dataset_name: str) -> Tuple[Path, Path]:
    """
    Ensure train/val CSVs for the given dataset exist and have data.
    If missing or empty, regenerate from raw audio files.

    Args:
        dataset_name: Dataset name (e.g. "Pitt", "Lu")

    Returns:
        Tuple[Path, Path]: (train_csv, val_csv)
    """
    raw_dir = PROJECT_ROOT / f"data/raw/{dataset_name}"
    train_csv = PROJECT_ROOT / f"data/processed/{dataset_name}-xlsr-train.csv"
    val_csv = PROJECT_ROOT / f"data/processed/{dataset_name}-xlsr-val.csv"
    feature_dir_name = f"{dataset_name}_xlsr_features"

    def csv_has_data(path):
        return path.exists() and path.stat().st_size > 30

    if not csv_has_data(train_csv) or not csv_has_data(val_csv):
        create_train_val_split(
            raw_audio_dir=raw_dir,
            dataset_name=dataset_name,
            feature_dir_name=feature_dir_name,
            xlsr=True
        )

    return train_csv, val_csv


def create_sub_split(
    source_dataset_name: str,
    dataset_name: str,
    feature_dir_name: str,
) -> Tuple[Path, Path]:
    """
    Create train/val CSVs for a dataset variant by remapping feature paths
    from the base dataset's split. Ensures all variants share the same
    train/val partition as the source dataset.

    Args:
        source_dataset_name: Base dataset name (e.g. "Pitt", "Lu")
        dataset_name: Variant name (e.g. "Pitt-Demucs", "Lu-Demucs")
        feature_dir_name: Feature directory name (e.g. "Pitt-Demucs_xlsr_features")

    Returns:
        Tuple[Path, Path]: (train_csv_path, val_csv_path)
    """
    source_train_csv, source_val_csv = create_split(source_dataset_name)

    train_csv = PROJECT_ROOT / f"data/processed/{dataset_name}-xlsr-train.csv"
    val_csv = PROJECT_ROOT / f"data/processed/{dataset_name}-xlsr-val.csv"

    for src_csv, dst_csv in [(source_train_csv, train_csv), (source_val_csv, val_csv)]:
        with open(src_csv, 'r') as f_in, open(dst_csv, 'w', newline='') as f_out:
            reader = csv.DictReader(f_in)
            writer = csv.writer(f_out)
            writer.writerow(['session_id', 'xlsr_path', 'ad'])
            for row in reader:
                session_id = row['session_id']
                feature_path = f"{feature_dir_name}/{session_id}.xlsr.pt"
                writer.writerow([session_id, feature_path, row['ad']])

    print(f"Created {train_csv.name} and {val_csv.name} from {source_dataset_name} raw split")
    return train_csv, val_csv


def create_test_csv(
    raw_audio_dir: Path,
    dataset_name: str,
    feature_dir_name: str,
    xlsr: bool = False,
) -> Path:
    """
    Create test CSV file containing all audio files
    
    Args:
        raw_audio_dir: Directory containing raw audio files 
                       (with subfolders Control and Dementia)
        dataset_name: Dataset name (used to construct CSV path)
        feature_dir_name: Feature directory name
        xlsr: Whether to use XLSR feature mode 
    
    Returns:
        Path: Test CSV path
    """
    
    feature_dir_name = str(feature_dir_name)
    
    # Build CSV path
    test_csv_path = PROJECT_ROOT / f"data/processed/{dataset_name}-xlsr-test.csv"
    test_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all audio files
    all_samples = []
    control_dir = raw_audio_dir / "Control"
    dementia_dir = raw_audio_dir / "Dementia"
    
    # Process Control group
    for audio_file in sorted(list(control_dir.glob("*.wav")) + list(control_dir.glob("*.mp3"))):
        all_samples.append({'session_id': audio_file.stem, 'ad': 0})
    
    # Process Dementia group
    for audio_file in sorted(list(dementia_dir.glob("*.wav")) + list(dementia_dir.glob("*.mp3"))):
        all_samples.append({'session_id': audio_file.stem, 'ad': 1})
    
    # Set column names and file extension
    feature_col = 'xlsr_path' if xlsr else 'feature_path'
    feature_ext = '.xlsr.pt' if xlsr else '.pt'
    
    # Generate test CSV
    with open(test_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', feature_col, 'ad'])
        for sample in all_samples:
            session_id = sample['session_id']
            feature_path = f"{feature_dir_name}/{session_id}{feature_ext}"
            writer.writerow([session_id, feature_path, sample['ad']])

    return test_csv_path
