# Back to the Original Pitt Corpus: The Hidden Cost of Speech-Enhanced Datasets for Alzheimer's Detection

Systematic study of how speech enhancement and dataset filtering affect Alzheimer's disease (AD) detection from spontaneous speech. We compare the raw Pitt-origin corpus against four processed variants (Pitt, ADReSS, ADReSSo, ADReSS-M) using three deep learning models and five audio-LLMs, with Lu as the cross-domain test set.

**Key finding.** Speech enhancement and sample filtering help in-domain metrics but hurt cross-domain generalization and shift LLM decision boundaries. Unprocessed Pitt-origin is the better training set for real-world AD speech detection.

## Layout

- `ad_detection/`
  - `train/` — three deep-learning models (`SLS_Model/`, `XLSR_model/`, `eGeMAPS_model/`), each with `config.py` / `extract_feature.py` / `model.py` / `train.py` / `test.py`; shared helpers in `utils/`.
  - `train_notebook/`
    - `SLS/` — training and testing notebooks for the SLS-based model.
    - `XLSR/` — training and testing notebooks for the XLSR-based model.
    - `eGeMAPS/` — training and testing notebooks for the eGeMAPS-based model.
    - `LLM/` — Zero-shot / two-shot audio-only evaluation of Kimi-Audio, Qwen2-Audio, Qwen3-Omni, Audio Flamingo 3, Ultravox.  Audio-only and audio+transcript evaluation for Limi-Audio
  - `data/` — dataset layout (raw / denoised / processed / transcripts). See `ad_detection/data/README.md`.
  - `models/` — checkpoints. See `ad_detection/models/Readme.md`.
- `requirements/` — `model_env/deep-requirements.txt` for the three deep models (the only env that needs fairseq); `LLMs_env/` has a separate env per audio-LLM.

## Datasets

| Name        | Processing                                        | Source                          |
|-------------|---------------------------------------------------|---------------------------------|
| Pitt-origin | Raw data                                          | [DementiaBank — Pitt-origin](https://talkbank.org/dementia/access/English/Pitt-orig.html) |
| Pitt        | Derived from Pitt-origin (denoised)               | [DementiaBank — Pitt](https://talkbank.org/dementia/access/English/Pitt.html)             |
| ADReSS      | Derived from Pitt-origin (filtered + enhanced)    | [DementiaBank — ADReSS (2020)](https://talkbank.org/dementia/ADReSS-2020/)            |
| ADReSSo     | Derived from Pitt-origin (filtered + enhanced)    | [DementiaBank — ADReSSo (2021)](https://talkbank.org/dementia/ADReSSo-2021/index.html) |
| ADReSS-M    | Derived from Pitt-origin (filtered, not enhanced) | [DementiaBank — ADReSS-M (2023)](https://media.talkbank.org/dementia/English/0extra/ADReSS-M) |
| Lu          | Raw data (distinct from Pitt-origin)              | [DementiaBank — Lu](https://talkbank.org/dementia/access/English/Lu.html)             |

All datasets from [DementiaBank](https://dementia.talkbank.org/). All five Pitt variants use the "Cookie Theft" picture description task.

## Alzheimer's Disease detection from Speech Challenges

The ADReSS / ADReSSo / ADReSS-M datasets were each released as part of a corresponding challenge:

- [ADReSS Challenge](https://luzs.gitlab.io/adress/) — Interspeech 2020.
- [ADReSSo Challenge](https://luzs.gitlab.io/adresso-2021/) — Interspeech 2021.
- [ADReSS-M Challenge](https://luzs.gitlab.io/madress-2023/) — ICASSP 2023.

## Frozen pretrained backbone

`xlsr2_300m.pt` — XLS-R-53 (300M) wav2vec 2.0 checkpoint, used as a frozen feature extractor in the SLS-based and XLSR-based models. 

Download From: <https://huggingface.co/facebook/wav2vec2-xls-r-300m>.

## Requirements

Create one conda env per requirements file. Each LLM uses its own env to avoid dependency conflicts.

- `requirements/model_env/deep-requirements.txt` — env `deep`, used by all three deep-learning models (SLS, XLSR, eGeMAPS).
- `requirements/LLMs_env/kimi-requirements.txt` — env for Kimi-Audio.
- `requirements/LLMs_env/qwen-requirements.txt` — env for Qwen2-Audio and Qwen3-Omni.
- `requirements/LLMs_env/audio-flamingo3-requirements.txt` — env for Audio Flamingo 3.
- `requirements/LLMs_env/ultravox-requirements.txt` — env for Ultravox.

## Setup

The fairseq is required **only by the `deep` env** (the LLM envs do not need it). After installing `deep-requirements.txt`, install fairseq into the `deep` env:

```bash
pip install -e ad_detection/train/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
```
