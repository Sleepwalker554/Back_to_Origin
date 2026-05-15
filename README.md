# Back to the Original Pitt Corpus: The Hidden Cost of Denoising Speech Datasets for Alzheimer’s Detection

Alzheimer's / dementia detection from spontaneous speech. Compares performance on raw vs. denoised audio across different audio-LLM settings (zero-shot / few-shot) and three deep learning-based models.

## Layout

- `ad_detection/` — XLSR baseline + audio-LLM evaluation.
  - `train/` — XLSR fine-tuning.
  - `train_notebook/` — training notebooks. `LLM/` holds audio-LLM evaluation notebooks (Kimi-Audio, Qwen2-Audio, Qwen3-Omni, Audio-Flamingo3, Ultravox), zero-shot and few-shot variants.
  - `data/` — data splits.
  - `models/` — checkpoints.
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

## Frozen pretrained backbone

- `xlsr2_300m.pt` — XLS-R-53 (300M params) wav2vec 2.0 checkpoint from
  fairseq. Used in `ad_detection/train/XLSR_model/model.py:SSLModel` as a frozen feature
  extractor (`freeze_xlsr=True`).
  Model page: <https://huggingface.co/facebook/wav2vec2-xls-r-300m>

## Setup

Install the vendored fairseq (pinned at commit `a54021305d`):

```bash
pip install -e ad_detection/train/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
```
