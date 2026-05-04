# Few-Shot is All You Need

Alzheimer's / dementia detection from spontaneous speech, comparing audio LLMs (zero-shot and few-shot) against a fine-tuned XLSR baseline, with optional speech-enhancement preprocessing.

## Layout

- `LLM/` — audio-LLM evaluation notebooks (Kimi-Audio, Qwen2-Audio, Qwen3-Omni, Audio-Flamingo3, Ultravox), zero-shot and few-shot variants. Outputs go to `LLM/results/<model>/` (gitignored).
- `ad_detection/` — XLSR fine-tuning baseline (`train/`), data splits (`data/`), checkpoints (`models/`).
- `Noise_Remove/` — speech-enhancement preprocessing (Demucs, Denoiser, FRCRN, MossFormer, Resemble).
- `Evaluate_Audio/` — audio quality metrics (DNSMOS) and spectrogram visualization.
- `dataset_analysis/` — duration / distribution stats over the corpora.
- `requirements/` — per-model dependency files; each LLM uses its own conda env.

## Datasets

Pitt, Pitt-origin, Lu, ADReSS, ADReSSo, ADReSS-M. Raw audio under `ad_detection/data/raw/<dataset>/{Control,Dementia}/` (gitignored).

## Usage

Pick the matching env from `requirements/` per model, then run the notebook. Results are written to `LLM/results/<model>/<dataset>-<denoising>.csv`.
