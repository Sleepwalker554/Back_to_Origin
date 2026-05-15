
## Model Architectures and Results

![Model Architectures](images/EMNLP_Model_Arc.png)

### **SLS-based Model**

The model uses `Sensitive Layer Selection (SLS)` on cached multi-layer `XLS-R` representations. Given frame-level representations from all transformer layers, the model first applies mask-aware mean pooling over time for each layer and predicts layer-wise weights through a linear layer followed by a sigmoid function. The original frame-level features are then aggregated across layers using the learned weights to obtain a weighted speech representation. This representation is passed through an classification head, including batch normalization, temporal average pooling, a one-dimensional convolutional layer, and attention pooling. Finally, a linear classification layer outputs the binary prediction.

### **XLSR-based Model**

The audio input is first fed into a pre-trained `XLS-R` model with frozen parameters to extract frame-level speech embeddings. The extracted features are then passed through a classification head similar to that of the SLS-based model, including batch normalization, a one-dimensional convolutional layer, and attention pooling, but without temporal average pooling. During attention pooling, a padding mask is applied to prevent padded frames from interfering with the results. Finally, a linear classification layer outputs the binary prediction.

### **eGeMAPS-based Model**

The model takes 25-dimensional `eGeMAPS` acoustic features extracted using the `OpenSMILE toolkit` as input. The features are first processed by two linear layers with batch normalization, ReLU activation, and dropout to remap the feature dimensions from 25 to 64 and then to 32. An attention pooling layer is then applied along the temporal dimension to aggregate frame-level representations into a fixed-dimensional vector. Finally, a linear classification layer outputs the AD prediction.

## Frozen pretrained backbone

XLS-R-53 (300M), used as a frozen feature extractor in the SLS-based and XLSR-based models. 

Download `xlsr2_300m.pt` From: <https://huggingface.co/facebook/wav2vec2-xls-r-300m> and place it under `ad_detection/models/`.

XLS-R requires `fairseq` (only in the `deep` env). After installing `deep-requirements.txt`, install `fairseq` in the `deep` env:

```bash
pip install -e ad_detection/train/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
```

## Requirements

All three deep learning-base models use the same `deep` environment.

- `requirements/model_env/deep-requirements.txt` — env `deep`, used by all three deep-learning models (SLS, XLSR, eGeMAPS).