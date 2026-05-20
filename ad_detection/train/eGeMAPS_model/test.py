import csv
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
from utils.config import PROJECT_ROOT
from utils.dataset import create_dataloaders
from .extract_feature import extract_egemaps_features_from_csv
from utils.data_split import create_test_csv


def test_on_dataset(dataset_name, model, device, raw_audio_dir):
    """
    Test eGeMAPS model on a given dataset.
    """
    print(f"\n{'='*60}")
    print(f"Testing on {dataset_name} Dataset")
    print(f"{'='*60}\n")

    # Build paths
    csv_path = PROJECT_ROOT / f"data/processed/{dataset_name}-egemaps-test.csv"
    audio_dir = PROJECT_ROOT / raw_audio_dir

    # Create CSV if it doesn't exist
    if not csv_path.exists():
        create_test_csv(
            raw_audio_dir=audio_dir,
            dataset_name=dataset_name,
            feature_dir_name=f"{dataset_name}_egemaps_features",
            xlsr=False,
        )

    # Extract features (function will skip existing files)
    extract_egemaps_features_from_csv(
        csv_path=csv_path,
        raw_audio_dir=audio_dir,
    )

    # Create test data loader
    test_loader = create_dataloaders(
        data_csv=csv_path,
        num_workers=0,
        xlsr=False,
    )

    # Perform predictions
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc=f"Testing on {dataset_name}"):
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, pos_label=1)

    control_mask = all_labels == 0
    dementia_mask = all_labels == 1
    control_acc = accuracy_score(all_labels[control_mask], all_preds[control_mask])
    dementia_acc = accuracy_score(all_labels[dementia_mask], all_preds[dementia_mask])

    print(f"Overall Accuracy: {accuracy*100:.2f}%, F1 Score: {f1:.4f}")
    print(f"Control Accuracy: {control_acc*100:.2f}%, Dementia Accuracy: {dementia_acc*100:.2f}%")

    return {
        'accuracy': accuracy,
        'f1': f1,
        'control_acc': control_acc,
        'dementia_acc': dementia_acc,
        'n_samples': len(all_labels),
        'n_control': np.sum(control_mask),
        'n_dementia': np.sum(dementia_mask),
    }


def test_on_dataset_with_val_csv(
    reference_val_csv: Path,
    dataset_name: str,
    model,
    device,
    raw_audio_dir: str,
):
    """
    Test eGeMAPS model on a dataset using the same validation set split as reference CSV.
    """
    print(f"\n{'='*60}")
    print(f"Testing on {dataset_name} Dataset (using reference VAL_CSV)")
    print(f"{'='*60}\n")

    target_val_csv = PROJECT_ROOT / f"data/processed/{dataset_name}-egemaps-val.csv"
    audio_dir = PROJECT_ROOT / raw_audio_dir

    target_val_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading reference VAL_CSV: {reference_val_csv}")
    reference_samples = []
    with open(reference_val_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            reference_samples.append({
                'session_id': row['session_id'],
                'ad': int(row['ad']),
            })

    print(f"Found {len(reference_samples)} samples in reference VAL_CSV")

    print(f"Checking audio file existence in {audio_dir}...")
    control_dir = audio_dir / "Control"
    dementia_dir = audio_dir / "Dementia"

    existing_samples = []
    missing_samples = []

    for sample in reference_samples:
        session_id = sample['session_id']
        ad = sample['ad']

        if ad == 0:
            audio_dir_to_check = control_dir
        else:
            audio_dir_to_check = dementia_dir

        wav_file = audio_dir_to_check / f"{session_id}.wav"
        mp3_file = audio_dir_to_check / f"{session_id}.mp3"

        if wav_file.exists() or mp3_file.exists():
            existing_samples.append(sample)
        else:
            missing_samples.append(session_id)

    print(f"Audio files found: {len(existing_samples)}/{len(reference_samples)}")
    if missing_samples:
        print(f"Missing audio files: {len(missing_samples)}")
        if len(missing_samples) <= 10:
            print(f"  Missing IDs: {', '.join(missing_samples)}")
    print()

    if len(existing_samples) == 0:
        raise ValueError(
            "No audio files matched the reference validation CSV under the target raw_audio_dir. "
            "session_id values in the reference CSV must exist as "
            "{session_id}.wav or {session_id}.mp3 under Control/ or Dementia/."
        )

    print(f"Creating target VAL_CSV: {target_val_csv}")
    with open(target_val_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', 'feature_path', 'ad'])
        for sample in existing_samples:
            session_id = sample['session_id']
            feature_path = f"{dataset_name}_egemaps_features/{session_id}.pt"
            ad = sample['ad']
            writer.writerow([session_id, feature_path, ad])

    print(f"Target VAL_CSV created with {len(existing_samples)} samples\n")

    extract_egemaps_features_from_csv(
        csv_path=target_val_csv,
        raw_audio_dir=audio_dir,
    )

    test_loader = create_dataloaders(
        data_csv=target_val_csv,
        num_workers=0,
        xlsr=False,
    )

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc=f"Testing on {dataset_name}"):
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, pos_label=1)

    control_mask = all_labels == 0
    dementia_mask = all_labels == 1
    control_acc = accuracy_score(all_labels[control_mask], all_preds[control_mask])
    dementia_acc = accuracy_score(all_labels[dementia_mask], all_preds[dementia_mask])

    print(f"Overall Accuracy: {accuracy*100:.2f}%, F1 Score: {f1:.4f}")
    print(f"Control Accuracy: {control_acc*100:.2f}%, Dementia Accuracy: {dementia_acc*100:.2f}%")

    return {
        'accuracy': accuracy,
        'f1': f1,
        'control_acc': control_acc,
        'dementia_acc': dementia_acc,
        'n_samples': len(all_labels),
        'n_control': np.sum(control_mask),
        'n_dementia': np.sum(dementia_mask),
    }
