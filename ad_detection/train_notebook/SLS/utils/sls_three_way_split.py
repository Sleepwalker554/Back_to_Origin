import csv
from pathlib import Path
from random import Random
from typing import Optional

from utils.config import PROJECT_ROOT, RANDOM_SEED
from utils.data_split import TAG_TO_COL, TAG_TO_EXT


def _collect_samples(raw_audio_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    control_dir = raw_audio_dir / "Control"
    dementia_dir = raw_audio_dir / "Dementia"

    if not control_dir.exists():
        raise FileNotFoundError(f"Control dir does not exist: {control_dir}")
    if not dementia_dir.exists():
        raise FileNotFoundError(f"Dementia dir does not exist: {dementia_dir}")

    control_samples = [
        {"session_id": audio_file.stem, "ad": 0}
        for audio_file in sorted(list(control_dir.glob("*.wav")) + list(control_dir.glob("*.mp3")))
    ]
    dementia_samples = [
        {"session_id": audio_file.stem, "ad": 1}
        for audio_file in sorted(list(dementia_dir.glob("*.wav")) + list(dementia_dir.glob("*.mp3")))
    ]

    if not control_samples:
        raise ValueError(f"No control audio files found in {control_dir}")
    if not dementia_samples:
        raise ValueError(f"No dementia audio files found in {dementia_dir}")

    return control_samples, dementia_samples


def _split_class_samples(
    samples: list[dict[str, object]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    n_total = len(samples)
    raw_counts = [
        n_total * train_ratio,
        n_total * val_ratio,
        n_total * test_ratio,
    ]
    counts = [int(count) for count in raw_counts]
    missing = n_total - sum(counts)
    remainders = sorted(
        range(len(raw_counts)),
        key=lambda idx: raw_counts[idx] - counts[idx],
        reverse=True,
    )
    for idx in remainders[:missing]:
        counts[idx] += 1

    n_train, n_val, _ = counts

    train_rows = samples[:n_train]
    val_rows = samples[n_train:n_train + n_val]
    test_rows = samples[n_train + n_val:]
    return train_rows, val_rows, test_rows


def _write_sls_csv(csv_path: Path, rows: list[dict[str, object]], feature_dir_name: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["session_id", TAG_TO_COL["sls"], "ad"])
        for row in rows:
            session_id = str(row["session_id"])
            writer.writerow([session_id, f"{feature_dir_name}/{session_id}{TAG_TO_EXT['sls']}", int(row["ad"])])


def create_sls_train_val_test_split(
    dataset_name: str,
    raw_audio_dir: Optional[Path] = None,
    feature_dir_name: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_seed: int = RANDOM_SEED,
) -> tuple[Path, Path, Path]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    raw_dir = raw_audio_dir or PROJECT_ROOT / f"data/raw/{dataset_name}"
    feature_dir = feature_dir_name or f"{dataset_name}_sls_features"

    train_csv = PROJECT_ROOT / f"data/processed/{dataset_name}-sls-70train.csv"
    val_csv = PROJECT_ROOT / f"data/processed/{dataset_name}-sls-10val.csv"
    test_csv = PROJECT_ROOT / f"data/processed/{dataset_name}-sls-20test.csv"

    control_samples, dementia_samples = _collect_samples(raw_dir)

    rng = Random(random_seed)
    rng.shuffle(control_samples)
    rng.shuffle(dementia_samples)

    control_train, control_val, control_test = _split_class_samples(
        control_samples, train_ratio, val_ratio, test_ratio
    )
    dementia_train, dementia_val, dementia_test = _split_class_samples(
        dementia_samples, train_ratio, val_ratio, test_ratio
    )

    train_rows = control_train + dementia_train
    val_rows = control_val + dementia_val
    test_rows = control_test + dementia_test

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    _write_sls_csv(train_csv, train_rows, feature_dir)
    _write_sls_csv(val_csv, val_rows, feature_dir)
    _write_sls_csv(test_csv, test_rows, feature_dir)

    print(f"============= {dataset_name} SLS Train(70%) / Val(10%) / Test(20%) Split Complete! =============")
    print(f"Training set: {len(train_rows)} samples (Control: {len(control_train)}, Dementia: {len(dementia_train)})")
    print(f"Validation set: {len(val_rows)} samples (Control: {len(control_val)}, Dementia: {len(dementia_val)})")
    print(f"Test set: {len(test_rows)} samples (Control: {len(control_test)}, Dementia: {len(dementia_test)})")

    return train_csv, val_csv, test_csv
