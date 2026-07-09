from __future__ import annotations

import csv
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
AD_ROOT = NOTEBOOK_DIR.parents[1]
TRAIN_DIR = AD_ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

PITT_INFO_CSV = NOTEBOOK_DIR / "pitt-origin-info" / "pitt_origin_info.csv"
PITT_RAW_AUDIO_DIR = AD_ROOT / "data" / "raw" / "Pitt-origin"
LU_RAW_AUDIO_DIR = AD_ROOT / "data" / "raw" / "Lu"
PROCESSED_DIR = AD_ROOT / "data" / "processed"

FEATURE_DIR_NAME = "Pitt-origin_sls_features"
PITT_SLS_FEATURES_DIR = PROCESSED_DIR / FEATURE_DIR_NAME

PROFILE_TABLES = {
    "ADReSS-like": [
        ("[50,55)", 2, 0, 2, 0),
        ("[55,60)", 7, 6, 7, 6),
        ("[60,65)", 4, 9, 4, 9),
        ("[65,70)", 9, 14, 9, 14),
        ("[70,75)", 9, 11, 9, 11),
        ("[75,80)", 4, 3, 4, 3),
    ],
    "ADReSSo-like": [
        ("[50,55)", 1, 0, 1, 1),
        ("[55,60)", 7, 6, 6, 16),
        ("[60,65)", 7, 8, 7, 13),
        ("[65,70)", 7, 22, 11, 23),
        ("[70,75)", 9, 21, 10, 18),
        ("[75,80]", 12, 22, 5, 4),
    ],
}


def _age_bin(age: int, profile_name: str) -> str | None:
    for lo, hi in [(50, 55), (55, 60), (60, 65), (65, 70), (70, 75)]:
        if lo <= age < hi:
            return f"[{lo},{hi})"
    if profile_name == "ADReSS-like" and 75 <= age < 80:
        return "[75,80)"
    if profile_name == "ADReSSo-like" and 75 <= age <= 80:
        return "[75,80]"
    return None


def _target_counts(profile_name: str) -> Counter:
    if profile_name not in PROFILE_TABLES:
        raise ValueError(f"Unknown profile_name: {profile_name}")

    counts = Counter()
    for age_bin, ad_male, ad_female, control_male, control_female in PROFILE_TABLES[profile_name]:
        counts[(age_bin, 1, "male")] = ad_male
        counts[(age_bin, 1, "female")] = ad_female
        counts[(age_bin, 0, "male")] = control_male
        counts[(age_bin, 0, "female")] = control_female
    return counts


def _audio_exists(session_id: str, ad: int) -> bool:
    folder = "Control" if ad == 0 else "Dementia"
    base = PITT_RAW_AUDIO_DIR / folder / session_id
    return base.with_suffix(".wav").exists() or base.with_suffix(".mp3").exists()


def load_pitt_candidates(profile_name: str) -> dict[tuple[str, int, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    with PITT_INFO_CSV.open("r", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            session_id = row["name"]
            label = row["label"].strip()
            sex = row["sex"].strip().lower()
            age_text = row["age"].strip()
            if label not in {"Control", "Dementia"} or sex not in {"male", "female"} or not age_text:
                continue
            ad = 0 if label == "Control" else 1
            if not _audio_exists(session_id, ad):
                continue
            age_bin = _age_bin(int(float(age_text)), profile_name)
            if age_bin is None:
                continue
            grouped[(age_bin, ad, sex)].append(
                {
                    "session_id": session_id,
                    "ad": ad,
                    "sex": sex,
                    "age": int(float(age_text)),
                    "age_bin": age_bin,
                }
            )

    for samples in grouped.values():
        samples.sort(key=lambda item: str(item["session_id"]))
    return grouped


def validate_profile_availability(profile_name: str) -> dict[str, object]:
    targets = _target_counts(profile_name)
    candidates = load_pitt_candidates(profile_name)
    shortages = []
    for key, needed in targets.items():
        available = len(candidates.get(key, []))
        if available < needed:
            shortages.append({"stratum": key, "needed": needed, "available": available})
    if shortages:
        raise ValueError(f"Not enough Pitt-origin samples for {profile_name}: {shortages}")

    return {
        "profile_name": profile_name,
        "target_total": sum(targets.values()),
        "candidate_total": sum(len(v) for v in candidates.values()),
        "target_counts": dict(targets),
    }


def _write_sls_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["session_id", "sls_path", "ad"])
        for row in rows:
            session_id = str(row["session_id"])
            writer.writerow([session_id, f"{FEATURE_DIR_NAME}/{session_id}.sls.pt", int(row["ad"])])


def _stratified_train_val_split(
    rows: list[dict[str, object]], train_ratio: float, rng: random.Random
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["age_bin"]), int(row["ad"]), str(row["sex"]))].append(row)

    train_rows = []
    val_rows = []
    for samples in grouped.values():
        samples = list(samples)
        rng.shuffle(samples)
        n_train = int(len(samples) * train_ratio)
        if len(samples) > 1:
            n_train = min(max(n_train, 1), len(samples) - 1)
        train_rows.extend(samples[:n_train])
        val_rows.extend(samples[n_train:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def create_profile_repeat_split(
    profile_name: str,
    repeat_idx: int,
    repeat_seed: int,
    train_ratio: float = 0.8,
) -> tuple[Path, Path, dict[str, object]]:
    targets = _target_counts(profile_name)
    candidates = load_pitt_candidates(profile_name)
    rng = random.Random(repeat_seed)

    sampled_rows: list[dict[str, object]] = []
    for key, needed in sorted(targets.items()):
        pool = list(candidates.get(key, []))
        if len(pool) < needed:
            raise ValueError(f"Stratum {key} needs {needed}, found {len(pool)}")
        sampled_rows.extend(rng.sample(pool, needed))

    train_rows, val_rows = _stratified_train_val_split(sampled_rows, train_ratio, rng)

    stem = f"Pitt-origin-{profile_name}-sls-repeat_{repeat_idx:02d}"
    train_csv = PROCESSED_DIR / f"{stem}-train.csv"
    val_csv = PROCESSED_DIR / f"{stem}-val.csv"
    _write_sls_csv(train_csv, train_rows)
    _write_sls_csv(val_csv, val_rows)

    summary = summarize_profile_rows(sampled_rows)
    summary.update(
        {
            "profile_name": profile_name,
            "repeat_idx": repeat_idx,
            "repeat_seed": repeat_seed,
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "train_size": len(train_rows),
            "val_size": len(val_rows),
            "sampled_rows": [
                {**row, "split": "train"} for row in train_rows
            ] + [
                {**row, "split": "val"} for row in val_rows
            ],
        }
    )
    return train_csv, val_csv, summary


def summarize_profile_rows(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(rows)
    counts = Counter((row["age_bin"], row["ad"], row["sex"]) for row in rows)
    return {
        "total": len(rows),
        "control": sum(1 for row in rows if int(row["ad"]) == 0),
        "dementia": sum(1 for row in rows if int(row["ad"]) == 1),
        "stratum_counts": dict(counts),
    }


def ensure_sls_features_for_split(train_csv: Path, val_csv: Path, device, ssl_model=None):
    from SLS_Model.extract_feature import extract_feature, extract_features_from_csv

    if ssl_model is None:
        return extract_feature(
            train_csv=train_csv,
            val_csv=val_csv,
            raw_audio_dir=PITT_RAW_AUDIO_DIR,
            sls_features_dir=PITT_SLS_FEATURES_DIR,
            device=device,
        )

    extract_features_from_csv(
        csv_path=train_csv,
        split_name="Controlled Train Set",
        raw_audio_dir=PITT_RAW_AUDIO_DIR,
        sls_features_dir=PITT_SLS_FEATURES_DIR,
        device=device,
        ssl_model=ssl_model,
    )
    extract_features_from_csv(
        csv_path=val_csv,
        split_name="Controlled Val Set",
        raw_audio_dir=PITT_RAW_AUDIO_DIR,
        sls_features_dir=PITT_SLS_FEATURES_DIR,
        device=device,
        ssl_model=ssl_model,
    )
    return ssl_model


def create_lu_sls_test_csv(
    dataset_name: str = "Lu",
    audio_dir: Optional[Path] = None,
    feature_dir_name: Optional[str] = None,
) -> Path:
    audio_dir = Path(audio_dir) if audio_dir is not None else LU_RAW_AUDIO_DIR
    feature_dir_name = feature_dir_name or f"{dataset_name}_sls_features"
    csv_path = PROCESSED_DIR / f"{dataset_name}-sls-test.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder, ad in (("Control", 0), ("Dementia", 1)):
        label_audio_dir = audio_dir / folder
        for audio_file in sorted(list(label_audio_dir.glob("*.wav")) + list(label_audio_dir.glob("*.mp3"))):
            rows.append((audio_file.stem, f"{feature_dir_name}/{audio_file.stem}.sls.pt", ad))

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["session_id", "sls_path", "ad"])
        writer.writerows(rows)
    return csv_path


def evaluate_lu_with_predictions(
    model,
    device,
    ssl_model,
    batch_size: int = 16,
    dataset_name: str = "Lu",
    audio_dir: Optional[Path] = None,
    feature_dir_name: Optional[str] = None,
) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from SLS_Model.extract_feature import extract_features_from_csv
    from XLSR_model.model import SSLModel
    from utils.dataset import FeatureDataset, sls_pad_mask

    if ssl_model is None:
        ssl_model = SSLModel(device, freeze_xlsr=True)

    audio_dir = Path(audio_dir) if audio_dir is not None else LU_RAW_AUDIO_DIR
    feature_dir_name = feature_dir_name or f"{dataset_name}_sls_features"
    csv_path = create_lu_sls_test_csv(
        dataset_name=dataset_name,
        audio_dir=audio_dir,
        feature_dir_name=feature_dir_name,
    )
    features_dir = PROCESSED_DIR / feature_dir_name
    extract_features_from_csv(
        csv_path=csv_path,
        split_name=f"{dataset_name} Test Set",
        raw_audio_dir=audio_dir,
        sls_features_dir=features_dir,
        device=device,
        ssl_model=ssl_model,
    )

    dataset = FeatureDataset(csv_path, feature_type="sls", lazy=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=sls_pad_mask,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels, masks in tqdm(loader, desc=f"Testing on {dataset_name}"):
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            logits = model(features, masks)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    session_ids = list(dataset.session_ids)
    metrics = compute_binary_metrics(all_labels, all_preds)
    return {
        **metrics,
        "n_samples": len(all_labels),
        "session_ids": session_ids,
        "y_true": all_labels,
        "y_pred": all_preds,
    }


def compute_binary_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    control_mask = y_true == 0
    dementia_mask = y_true == 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "control_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "dementia_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "control_acc": float(accuracy_score(y_true[control_mask], y_pred[control_mask]))
        if control_mask.any()
        else math.nan,
        "dementia_acc": float(accuracy_score(y_true[dementia_mask], y_pred[dementia_mask]))
        if dementia_mask.any()
        else math.nan,
    }


def bootstrap_binary_metrics(
    y_true,
    y_pred,
    n_bootstrap: int = 1000,
    sample_size: int | None = None,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    summary, _ = bootstrap_binary_metrics_with_samples(
        y_true=y_true,
        y_pred=y_pred,
        n_bootstrap=n_bootstrap,
        sample_size=sample_size,
        seed=seed,
    )
    return summary


def bootstrap_binary_metrics_with_samples(
    y_true,
    y_pred,
    n_bootstrap: int = 1000,
    sample_size: int | None = None,
    seed: int = 0,
) -> tuple[dict[str, dict[str, float]], list[dict[str, float]]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if sample_size is None:
        sample_size = len(y_true)
    rng = np.random.default_rng(seed)
    values = defaultdict(list)
    sample_rows = []

    for bootstrap_idx in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=sample_size)
        metrics = compute_binary_metrics(y_true[indices], y_pred[indices])
        sample_row = {
            "bootstrap_idx": bootstrap_idx,
            "sample_size": sample_size,
            **metrics,
        }
        sample_rows.append(sample_row)
        for key, value in metrics.items():
            values[key].append(value)

    summary = {}
    for key, metric_values in values.items():
        arr = np.asarray(metric_values, dtype=float)
        arr = arr[~np.isnan(arr)]
        summary[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }
    return summary, sample_rows


def flatten_bootstrap_summary(summary: dict[str, dict[str, float]], prefix: str = "bootstrap") -> dict[str, float]:
    flat = {}
    for metric, stats in summary.items():
        for stat_name, value in stats.items():
            flat[f"{prefix}_{metric}_{stat_name}"] = value
    return flat


def save_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def summarize_metric_rows(rows: list[dict[str, object]], metric_keys: list[str]) -> dict[str, float]:
    summary = {}
    for key in metric_keys:
        arr = np.asarray([float(row[key]) for row in rows], dtype=float)
        summary[f"{key}_mean"] = float(np.mean(arr))
        summary[f"{key}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return summary
