## Datasets

| Name        | Source                                   |
|-------------|------------------------------------------|
| Pitt        | DementiaBank — Pitt                      |
| Pitt-origin | DementiaBank — Pitt-origin               |
| Lu          | DementiaBank — Lu                        |
| ADReSS      | DementiaBank — ADReSS challenge (2020)   |
| ADReSSo     | DementiaBank — ADReSSo challenge (2021)  |
| ADReSS-M    | DementiaBank — ADReSS-M challenge (2023) |

Download all the datasets from [DementiaBank](https://dementia.talkbank.org/) and place them as below.

## Expected layout

```
data/
├── raw/                              # Original audio per dataset
│   ├── Pitt/
│   │   ├── Control/
│   │   └── Dementia/
│   ├── Pitt-origin/
│   │   ├── Control/
│   │   └── Dementia/
│   ├── Lu/
│   │   ├── Control/
│   │   └── Dementia/
│   ├── ADReSS/
│   │   ├── Control/
│   │   └── Dementia/
│   ├── ADReSSo/
│   │   ├── Control/
│   │   └── Dementia/
│   └── ADReSS-M/
│       ├── Control/
│       └── Dementia/
│
├── denoised/                         # Denoised audio (output of Noise_Remove/)
│   ├── <dataset>-Demucs/
│   │   ├── Control/
│   │   └── Dementia/
│   ├── <dataset>-Denoiser/
│   │   ├── Control/
│   │   └── Dementia/
│   ├── <dataset>-FRCRN_SE/
│   │   ├── Control/
│   │   └── Dementia/
│   ├── <dataset>-MossFormer/
│   │   ├── Control/
│   │   └── Dementia/
│   └── <dataset>-Resemble/
│       ├── Control/
│       └── Dementia/
│
├── processed/                        # XLSR-extracted features + split CSVs
│
└── Text/                             # Transcripts
    ├── Pitt_Transcript/
    └── Lu_Transcript/
```
