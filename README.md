# Back to the Original Pitt Corpus: The Hidden Cost of Denoising Speech Datasets for Alzheimer’s Detection

Alzheimer's / dementia detection from spontaneous speech. Compares performance on raw vs. denoised audio across different audio-LLM settings (zero-shot / few-shot) and three deep learning-based models.

## Layout

- `ad_detection/train_notebook/LLM/` — audio-LLM evaluation notebooks (Kimi-Audio, Qwen2-Audio, Qwen3-Omni, Audio-Flamingo3, Ultravox), zero-shot and few-shot variants.
- `ad_detection/` — XLSR fine-tuning baseline (`train/`), data splits (`data/`), checkpoints (`models/`).
- `Noise_Remove/` — speech-enhancement preprocessing (Demucs, Denoiser, FRCRN, MossFormer, Resemble).
- `Evaluate_Audio/` — audio quality metrics (DNSMOS) and spectrogram visualization.
- `dataset_analysis/` — duration / distribution stats over the datasets.
- `requirements/` — each LLM uses its own conda env.

## Datasets

| Name        | Source                                   |
|-------------|------------------------------------------|
| Pitt        | DementiaBank — Pitt                      |
| Pitt-origin | DementiaBank — Pitt-origin               |
| Lu          | DementiaBank — Lu                        |
| ADReSS      | DementiaBank — ADReSS challenge (2020)   |
| ADReSSo     | DementiaBank — ADReSSo challenge (2021)  |
| ADReSS-M    | DementiaBank — ADReSS-M challenge (2023) |

Download all the datasets from [DementiaBank](https://dementia.talkbank.org/).
