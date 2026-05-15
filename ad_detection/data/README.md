## Datasets

| Name        | Processing                                        | Source                          |
|-------------|---------------------------------------------------|---------------------------------|
| Pitt-origin | Raw data                                          | [DementiaBank — Pitt-origin](https://talkbank.org/dementia/access/English/Pitt-orig.html) |
| Pitt        | Derived from Pitt-origin (denoised)               | [DementiaBank — Pitt](https://talkbank.org/dementia/access/English/Pitt.html)             |
| ADReSS      | Derived from Pitt-origin (filtered + enhanced)    | [DementiaBank — ADReSS (2020)](https://talkbank.org/dementia/ADReSS-2020/)            |
| ADReSSo     | Derived from Pitt-origin (filtered + enhanced)    | [DementiaBank — ADReSSo (2021)](https://talkbank.org/dementia/ADReSSo-2021/index.html) |
| ADReSS-M    | Derived from Pitt-origin (filtered, not enhanced) | [DementiaBank — ADReSS-M (2023)](https://luzs.gitlab.io/madress-2023/) |
| Lu          | Raw data (distinct from Pitt-origin)              | [DementiaBank — Lu](https://talkbank.org/dementia/access/English/Lu.html)             |

All datasets from [DementiaBank](https://dementia.talkbank.org/). All the datasets use the "Cookie Theft" picture description task.

| Dataset     | Total | AD  | Control |
|-------------|:-----:|:---:|:-------:|
| Pitt-origin |  552  | 309 |   243   |
| Pitt        |  551  | 309 |   242   |
| ADReSS      |  156  |  78 |    78   |
| ADReSSo     |  237  | 122 |   115   |
| ADReSS-M    |  237  | 122 |   115   |

*Sample numbers of datasets.*

| Dataset     | Min(s) | Max(s) | Mean(s) | Median(s) | Total(min) |
|-------------|:------:|:------:|:-------:|:---------:|:----------:|
| Pitt Corpus | 18.00  | 268.48 |  70.06  |   63.29   |   643.39   |
| Pitt-origin | 17.89  | 268.49 |  69.97  |   63.20   |   643.72   |
| ADReSS      | 26.06  | 268.49 |  75.30  |   70.05   |   195.78   |
| ADReSSo     | 22.35  | 268.49 |  76.89  |   70.47   |   303.72   |
| ADReSS-M    | 22.35  | 268.49 |  76.89  |   70.48   |   303.72   |

*Audio duration statistics per dataset.*

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
