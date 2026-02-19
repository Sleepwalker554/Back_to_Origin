import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from dataset import create_dataloaders
from extract_XLSR_feature import extract_features_from_csv


def test_on_dataset(dataset_name, csv_path, model, device, ssl_model, PROJECT_ROOT):
    """
    Test model on a given dataset
    
    Args:
        dataset_name: Name of the dataset (e.g., 'Pitt', 'Lu', 'ADReSS')
        csv_path: Path to the CSV file
        model: Model to test
        device: Device to run on
        ssl_model: SSL model for feature extraction
        PROJECT_ROOT: Project root path
    
    Returns:
        dict: Dictionary containing test results
    """
    print(f"\n{'='*60}")
    print(f"Testing on {dataset_name} Dataset")
    print(f"{'='*60}\n")
    
    # Configure paths
    features_dir = PROJECT_ROOT / f"data/processed/{dataset_name}_xlsr_features"
    audio_dir = PROJECT_ROOT / f"data/raw/{dataset_name}"
    
    # Extract features if needed
    if not features_dir.exists():
        print(f"Extracting XLSR features for {dataset_name}...")
        extract_features_from_csv(
            csv_path=csv_path,
            split_name=f"{dataset_name} Test Set",
            raw_audio_dir=audio_dir,
            xlsr_features_dir=features_dir,
            device=device,
            ssl_model=ssl_model,
        )

    # Create test data loader
    test_loader = create_dataloaders(
        data_csv=csv_path,
        batch_size=32,
        num_workers=0,
        xlsr=True
    )
    
    # Perform predictions
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
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, pos_label=1)
    
    # Calculate accuracy for each class
    control_mask = all_labels == 0
    dementia_mask = all_labels == 1
    control_acc = accuracy_score(all_labels[control_mask], all_preds[control_mask])
    dementia_acc = accuracy_score(all_labels[dementia_mask], all_preds[dementia_mask])
    
    # Print results
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
