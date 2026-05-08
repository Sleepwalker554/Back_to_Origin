# Models

This directory stores all model artifacts used by `ad_detection/`. Contents
are gitignored except for this README — `.pt` / checkpoint files are not
versioned (size-prohibitive); regenerate or download them per the sections
below.

## Frozen pretrained backbone

- `xlsr2_300m.pt` — XLS-R-53 (300M params) wav2vec 2.0 checkpoint from
  fairseq. Used in `train/XLSR_model/model.py:SSLModel` as a frozen feature
  extractor (`freeze_xlsr=True`).
  Download: https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt

## Auxiliary models

- `speech_frcrn_ans_cirm_16k/` — FRCRN speech-enhancement checkpoint used
  by `Noise_Remove/` to produce the `*-FRCRN_SE` denoised dataset variants.

## Trained AD-detection checkpoints

Each subdirectory `<dataset>_<feature>_multi_seed/` (or `_svm/`) holds the
best-of-seed models produced by training notebooks under `train_notebook/`.

Naming convention:

- prefix: dataset variant (e.g. `Pitt`, `Pitt-Demucs`, `Pitt-origin-Denoiser`)
- middle: feature backend (`xlsr`, `egemaps`)
- suffix: training scheme (`multi_seed`, `svm`)

Inside each: `seed_<N>/best.pth` — best-validation-accuracy checkpoint per
seed.
