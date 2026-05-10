import csv
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
from .config import NUM_WORKERS
from .data_split import TAG_TO_COL
# Per-model pad lengths live in each model's config to keep cross-cutting
# constants out of utils/config.py.
from XLSR_model.config import MAX_TIME_STEPS as XLSR_MAX_TIME_STEPS
from SLS_Model.config import MAX_TIME_STEPS as SLS_MAX_TIME_STEPS


def _resolve_feature_type(feature_type: Optional[str], xlsr: Optional[bool]) -> str:
    """Backward-compat: if caller passed xlsr=True/False, translate to feature_type."""
    if xlsr is not None:
        return 'xlsr' if xlsr else 'egemaps'
    return feature_type or 'egemaps'


class FeatureDataset(Dataset):
    """
    Load data from CSV.

    Supports three feature_type values:
      - 'egemaps': 2-D tensor (T, 25)        — small, preloaded to RAM
      - 'xlsr':   2-D tensor (T, 1024)       — ~12 MB / sample, preloaded
      - 'sls':    3-D tensor (L, T, 1024)    — ~140 MB / sample (fp16),
                                                lazy-loaded by default

    `lazy=True` skips preload — paths are stored and `torch.load` runs
    inside `__getitem__`. NaN/Inf validation is also deferred to load
    time (a corrupted file will raise from `__getitem__` instead of
    being silently skipped).
    """
    FEATURE_NAME_MAP = {'egemaps': 'eGeMAPS', 'xlsr': 'XLSR', 'sls': 'SLS'}

    def __init__(
            self,
            csv_path: Path,
            feature_type: Optional[str] = None,
            xlsr: Optional[bool] = None,
            lazy: Optional[bool] = None,
    ):
        super().__init__()

        self.csv_path = csv_path
        self.feature_type = _resolve_feature_type(feature_type, xlsr)
        if self.feature_type not in TAG_TO_COL:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

        self.feature_path_key = TAG_TO_COL[self.feature_type]
        self.feature_name = self.FEATURE_NAME_MAP[self.feature_type]

        # SLS cached features are ~140 MB each; preloading a Pitt-sized split
        # would consume ~75 GB of RAM. Default to lazy for SLS only.
        self.lazy = lazy if lazy is not None else (self.feature_type == 'sls')

        self.feature_paths = []  # always populated
        self.features = []        # populated only when not lazy
        self.labels = []
        self.session_ids = []

        self._load_data()

    def _load_data(self):
        """Index the CSV; optionally preload features into memory."""
        csv_dir = Path(self.csv_path).parent

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                session_id = row['session_id']
                feature_path = row[self.feature_path_key]
                ad = int(row['ad'])

                feature_path_abs = (csv_dir / feature_path).resolve()

                if not feature_path_abs.exists():
                    print(f"Error: {self.feature_name} feature file does not exist: {feature_path_abs}")
                    continue

                if self.lazy:
                    self.feature_paths.append(feature_path_abs)
                    self.labels.append(ad)
                    self.session_ids.append(session_id)
                    continue

                try:
                    features = torch.load(feature_path_abs)

                    if torch.isnan(features).any() or torch.isinf(features).any():
                        print(f"Error: Features contain NaN/Inf, skipping: {session_id}")
                        continue

                    self.features.append(features)
                    self.feature_paths.append(feature_path_abs)
                    self.labels.append(ad)
                    self.session_ids.append(session_id)

                except Exception as e:
                    print(f"Error: Loading features failed {session_id}: {e}")
                    continue

        num_control = sum(1 for label in self.labels if label == 0)
        num_dementia = sum(1 for label in self.labels if label == 1)

        if len(self.labels) == 0:
            print(f"Error: No {self.feature_name} feature files found!")
            raise ValueError("Dataset is empty.")

        mode = "lazy" if self.lazy else "preloaded"
        print(f"Loading completed ({mode}): {len(self)} samples")
        print(f"Control: {num_control}, Dementia: {num_dementia}")
        print("\n")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.lazy:
            features = torch.load(self.feature_paths[index])
        else:
            features = self.features[index]
        label = self.labels[index]
        return features, label

    def get_session_id(self, index):
        return self.session_ids[index]


def xlsr_pad_mask(batch):
    """Pad XLSR features (T, 1024) along T to XLSR_MAX_TIME_STEPS with attention mask."""
    features_list = [f for f, l in batch]
    labels_list = [l for f, l in batch]

    padded_features = []
    masks = []

    for features in features_list:
        seq_len = features.shape[0]

        mask = torch.ones(XLSR_MAX_TIME_STEPS)

        if seq_len > XLSR_MAX_TIME_STEPS:
            features = features[:XLSR_MAX_TIME_STEPS]
        elif seq_len < XLSR_MAX_TIME_STEPS:
            padding = torch.zeros(XLSR_MAX_TIME_STEPS - seq_len, features.shape[1])
            features = torch.cat([features, padding], dim=0)
            mask[seq_len:] = 0

        padded_features.append(features)
        masks.append(mask)

    features_batch = torch.stack(padded_features, dim=0)  # (Batch, MaxTime, 1024)
    labels_batch = torch.tensor(labels_list, dtype=torch.long)
    masks_batch = torch.stack(masks, dim=0)  # (Batch, MaxTime)

    return features_batch, labels_batch, masks_batch


def sls_pad_mask(batch):
    """
    Pad SLS features (L, T, 1024) along T (dim=1) to SLS_MAX_TIME_STEPS.

    Cached features are stored as fp16; cast to fp32 here so the model
    receives fp32 tensors regardless of disk dtype.
    """
    features_list = [f for f, l in batch]
    labels_list = [l for f, l in batch]

    padded_features = []
    masks = []

    for features in features_list:
        # Keep dtype as-is (fp16 on disk); cast to fp32 happens on GPU in train loop.
        L, seq_len, D = features.shape

        mask = torch.ones(SLS_MAX_TIME_STEPS)

        if seq_len > SLS_MAX_TIME_STEPS:
            features = features[:, :SLS_MAX_TIME_STEPS, :]
        elif seq_len < SLS_MAX_TIME_STEPS:
            padding = torch.zeros(L, SLS_MAX_TIME_STEPS - seq_len, D, dtype=features.dtype)
            features = torch.cat([features, padding], dim=1)
            mask[seq_len:] = 0

        padded_features.append(features)
        masks.append(mask)

    features_batch = torch.stack(padded_features, dim=0)  # (Batch, L, MaxTime, 1024)
    labels_batch = torch.tensor(labels_list, dtype=torch.long)
    masks_batch = torch.stack(masks, dim=0)  # (Batch, MaxTime)

    return features_batch, labels_batch, masks_batch


_COLLATE_BY_TYPE = {
    'egemaps': None,
    'xlsr': xlsr_pad_mask,
    'sls': sls_pad_mask,
}


def create_dataloaders(
        data_csv: Path,
        batch_size: int = 16,
        num_workers: int = NUM_WORKERS,
        feature_type: Optional[str] = None,
        xlsr: Optional[bool] = None,
        lazy: Optional[bool] = None,
):
    """
    Args:
        data_csv: Dataset set CSV path
        batch_size: Batch size
        num_workers: Number of worker processes
        feature_type: 'egemaps', 'xlsr', or 'sls' (default: 'egemaps')
        xlsr: Deprecated. True -> 'xlsr', False -> 'egemaps'.
        lazy: If True, FeatureDataset stores paths and loads features
              inside __getitem__. If None, defaults to True for 'sls'
              and False for 'xlsr' / 'egemaps'.

    Returns:
        data_loader: DataLoader
    """
    feature_type = _resolve_feature_type(feature_type, xlsr)
    if feature_type not in _COLLATE_BY_TYPE:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    try:
        dataset = FeatureDataset(data_csv, feature_type=feature_type, lazy=lazy)
    except ValueError as e:
        print(f"\nError: Loading data set failed: {e}")
        raise

    if len(dataset) == 0:
        raise ValueError(f"\nError: Dataset is empty!")

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_COLLATE_BY_TYPE[feature_type],
    )
    # SLS features are ~24x larger than XLSR; reduce worker prefetch to fit
    # 32GB CPU RAM. Other feature types keep PyTorch defaults.
    if feature_type == 'sls':
        loader_kwargs['num_workers'] = min(num_workers, 2)
        loader_kwargs['prefetch_factor'] = 1 if loader_kwargs['num_workers'] > 0 else None
        loader_kwargs['persistent_workers'] = False

    data_loader = DataLoader(dataset, **loader_kwargs)

    return data_loader
