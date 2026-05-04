from pathlib import Path

# ====== Paths ======
DATA_DIR     = Path("/root/autodl-tmp/Few-Shot_is_all_you_need/ad_detection/data")
RAW_DIR      = DATA_DIR / "raw"
DENOISED_DIR = DATA_DIR / "denoised"

DATASETS = ["Pitt-origin", "Lu"]


def get_audio_files(dataset_name):
    """Return {'Control': [files], 'Dementia': [files]} for a dataset."""
    raw = RAW_DIR / dataset_name
    return {
        label: list((raw / label).glob("*.wav")) + list((raw / label).glob("*.mp3"))
        for label in ("Control", "Dementia")
    }


def get_output_dir(dataset_name, method):
    """Return denoised output dir for (dataset, method)."""
    return DENOISED_DIR / f"{dataset_name}-{method}"
