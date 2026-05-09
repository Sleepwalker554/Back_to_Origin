import csv
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
from utils.config import PROJECT_ROOT
from utils.dataset import create_dataloaders
from .extract_feature import extract_features_from_csv
from utils.data_split import create_test_csv


def test_on_dataset(dataset_name, model, device, ssl_model, raw_audio_dir):
    """
    Test SLS model on a given dataset.

    Args:
        dataset_name: Name of the dataset
        model: AD_SLS_Model to test
        device: Device to run on
        ssl_model: SSL model for feature extraction
        raw_audio_dir: Path to raw audio directory (relative to PROJECT_ROOT)

    Returns:
        dict: test results
    """
    print(f"\n{'='*60}")
    print(f"Testing on {dataset_name} Dataset")
    print(f"{'='*60}\n")

    csv_path = PROJECT_ROOT / f"data/processed/{dataset_name}-sls-test.csv"
    features_dir = PROJECT_ROOT / f"data/processed/{dataset_name}_sls_features"
    audio_dir = PROJECT_ROOT / raw_audio_dir

    if not csv_path.exists():
        create_test_csv(
            raw_audio_dir=audio_dir,
            dataset_name=dataset_name,
            feature_dir_name=f"{dataset_name}_sls_features",
            feature_type='sls',
        )

    extract_features_from_csv(
        csv_path=csv_path,
        split_name=f"{dataset_name} Test Set",
        raw_audio_dir=audio_dir,
        sls_features_dir=features_dir,
        device=device,
        ssl_model=ssl_model,
    )

    test_loader = create_dataloaders(
        data_csv=csv_path,
        num_workers=0,
        feature_type='sls',
    )

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels, masks in tqdm(test_loader, desc=f"Testing on {dataset_name}"):
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            logits = model(features, masks)
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
        'n_dementia': np.sum(dementia_mask)
    }


def test_on_dataset_with_val_csv(
    reference_val_csv: Path,
    dataset_name: str,
    model,
    device,
    ssl_model,
    raw_audio_dir: str
):
    """
    Test SLS model on a dataset using the same validation set split as a
    reference CSV (for cross-corpus evaluation on shared session_ids).
    """
    print(f"\n{'='*60}")
    print(f"Testing on {dataset_name} Dataset (using reference VAL_CSV)")
    print(f"{'='*60}\n")

    target_val_csv = PROJECT_ROOT / f"data/processed/{dataset_name}-sls-val.csv"
    features_dir = PROJECT_ROOT / f"data/processed/{dataset_name}_sls_features"
    audio_dir = PROJECT_ROOT / raw_audio_dir

    target_val_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading reference VAL_CSV: {reference_val_csv}")
    reference_samples = []
    with open(reference_val_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            reference_samples.append({
                'session_id': row['session_id'],
                'ad': int(row['ad'])
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
        writer.writerow(['session_id', 'sls_path', 'ad'])
        for sample in existing_samples:
            session_id = sample['session_id']
            feature_path = f"{dataset_name}_sls_features/{session_id}.sls.pt"
            ad = sample['ad']
            writer.writerow([session_id, feature_path, ad])

    print(f"Target VAL_CSV created with {len(existing_samples)} samples\n")

    extract_features_from_csv(
        csv_path=target_val_csv,
        split_name=f"{dataset_name} Validation Set",
        raw_audio_dir=audio_dir,
        sls_features_dir=features_dir,
        device=device,
        ssl_model=ssl_model,
    )

    test_loader = create_dataloaders(
        data_csv=target_val_csv,
        num_workers=0,
        feature_type='sls',
    )

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels, masks in tqdm(test_loader, desc=f"Testing on {dataset_name}"):
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            logits = model(features, masks)
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
        'n_dementia': np.sum(dementia_mask)
    }
