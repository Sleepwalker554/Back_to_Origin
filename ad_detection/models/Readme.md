# Models

This directory stores all model artifacts used by `ad_detection/`.

## Frozen pretrained backbone

- `xlsr2_300m.pt` — XLS-R-53 (300M params) wav2vec 2.0 checkpoint from
  fairseq. Used in `train/XLSR_model/model.py:SSLModel` as a frozen feature
  extractor (`freeze_xlsr=True`).
  Model page: <https://huggingface.co/facebook/wav2vec2-xls-r-300m>
