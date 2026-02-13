import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from config import FEAT_SEQ_LEN, PROJECT_ROOT, BATCH_SIZE, NUM_WORKERS, XLSR_MAX_TIME_STEPS, XLSR_DIM_INPUT

class FeatureDataset(Dataset):
    """
    Load data from CSV, preload features to memory
    """
    def __init__(
            self,
            csv_path: Path,
            xlsr: bool = False,
    ):
        """
        Args:
            csv_path: CSV file path
            xlsr: True to use XLSR features, False to use eGeMAPS features
        """
        super().__init__()

        self.csv_path = csv_path
        self.xlsr = xlsr

        if xlsr:
            self.feature_path_key = 'xlsr_path'
            self.feature_name = 'XLSR'
            # self.expected_shape = (XLSR_SEGMENT_LEN, XLSR_DIM_INPUT)
        else:
            self.feature_path_key = 'egemaps_path'
            self.feature_name = 'eGeMAPS'
            self.expected_shape = (FEAT_SEQ_LEN, 25)

        # Store data
        self.features = []  # features
        self.labels = []  # AD labels (0 or 1)
        self.session_ids = []  # session_id

        # Load CSV and preload features
        self._load_data()

    def _load_data(self):
        """
        Load data from CSV and preload all features to memory
        """
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                session_id = row['session_id']
                feature_path = row[self.feature_path_key]
                ad = int(row['ad'])

                # Determine the path based on the feature type
                if self.xlsr:
                    csv_dir = Path(self.csv_path).parent
                    feature_path_abs = (csv_dir / feature_path).resolve()
                else:
                    feature_path_abs = PROJECT_ROOT / feature_path

                # Check if file exists
                if not feature_path_abs.exists():
                    print(f"Error: {self.feature_name} feature file does not exist: {feature_path_abs}")
                    continue

                # Load features
                try:
                    features = torch.load(feature_path_abs)
                    
                    # Check for NaN and Inf
                    if torch.isnan(features).any() or torch.isinf(features).any():
                        print(f"Error: Features contain NaN/Inf, skipping: {session_id}")
                        continue
                     
                    # Verify shape
                    # if features.shape != self.expected_shape:
                    #     print(f"Error: Feature shape error {session_id}: {features.shape}, expected {self.expected_shape}")
                    #     continue

                    # Store data
                    self.features.append(features)
                    self.labels.append(ad)
                    self.session_ids.append(session_id)

                except Exception as e:
                    print(f"Error: Loading features failed {session_id}: {e}")
                    continue

        num_control = sum(1 for label in self.labels if label == 0)
        num_dementia = sum(1 for label in self.labels if label == 1)

        if len(self.features) == 0:
            print(f"Error: No {self.feature_name} feature files found!")
            raise ValueError("Dataset is empty.")

        print(f"Loading completed: {len(self)} samples")
        print(f"Control: {num_control}, Dementia: {num_dementia}")
        print("\n")
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        """
        Args:
            index: sample index

        Returns:
            features: eGeMAPS features (FEAT_SEQ_LEN, 25) or XLSR features (XLSR_SEGMENT_LEN, XLSR_FEATURE_DIM)
            label: 0 (Control) or 1 (Dementia)
        """
        features = self.features[index]
        label = self.labels[index]

        return features, label

    def get_session_id(self, index):
        return self.session_ids[index]


def xlsr_pad_mask(batch):
    """Padding XLSR features to fixed length with attention mask"""
    features_list = [f for f, l in batch]
    labels_list = [l for f, l in batch]
    
    padded_features = []
    masks = []  # Attention mask: 1 for real data, 0 for padding
    
    for features in features_list:
        seq_len = features.shape[0]
        
        # Create mask: 1 for real data, 0 for padding
        mask = torch.ones(XLSR_MAX_TIME_STEPS)
        
        if seq_len > XLSR_MAX_TIME_STEPS:
            features = features[:XLSR_MAX_TIME_STEPS]
        elif seq_len < XLSR_MAX_TIME_STEPS:
            padding = torch.zeros(XLSR_MAX_TIME_STEPS - seq_len, features.shape[1])
            features = torch.cat([features, padding], dim=0)
            mask[seq_len:] = 0  # Mark padding positions as 0
        
        padded_features.append(features)
        masks.append(mask)
    
    features_batch = torch.stack(padded_features, dim=0)  # (Batch, MaxTime, 1024)
    labels_batch = torch.tensor(labels_list, dtype=torch.long)
    masks_batch = torch.stack(masks, dim=0)  # (Batch, MaxTime)
    
    return features_batch, labels_batch, masks_batch


def create_dataloaders(
        data_csv: Path,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        xlsr: bool = False,
):
    """
    Args:
        data_csv: Dataset set CSV path
        batch_size: Batch size
        num_workers: Number of worker processes
        xlsr: True to use XLSR features, False to use eGeMAPS features

    Returns:
        data_loader: DataLoader
    """
    feature_name = "XLSR" if xlsr else "eGeMAPS"

    # Create Dataset
    try:
        dataset = FeatureDataset(data_csv, xlsr=xlsr)
    except ValueError as e:
        print(f"\nError: Loading data set failed: {e}")
        raise

    # Check if datasets are empty
    if len(dataset) == 0:
        raise ValueError(f"\nError: Dataset is empty!")

    # Create DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training set
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=torch.cuda.is_available(),  # Speed up GPU transfer
        collate_fn=xlsr_pad_mask if xlsr else None,  # Padding for XLSR features
    )

    return data_loader
