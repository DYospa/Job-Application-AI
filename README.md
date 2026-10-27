# JobAI: Automated Job Application System

Capstone project (Yannik Alinyoh, Christo Antony, Doron Yospa, Anuraag Dasari).

JobAI reads a user's resume, screens out fraudulent and likely ghost job postings, ranks jobs by
how well they match the resume, and helps fill in applications, with the user reviewing everything
before it is submitted.

## Where things are

```
├── README.md                  this file
├── requirements.txt           packages for the data + model code
├── data/
│   ├── README.md              where to get the raw data, links to the processed data
│   └── v1/                    first merged dataset (Doron), kept for reference
├── src/
│   ├── data/
│   │   ├── flatten_techmap.py       Techmap 4.4 GB JSON -> flat CSV
│   │   ├── merge_linkedin_tables.py LinkedIn postings + lookup tables -> one table
│   │   └── build_dataset.py         builds the v2 ML-ready fraud dataset
│   ├── models/
│   │   └── train_baseline.py        baselines + metrics, logged to Weights & Biases
│   └── v1/                    first pipeline (merge, balance, clean), kept for reference
├── notebooks/
│   ├── 01_eda_checkpoint1.ipynb     EDA on raw LinkedIn + Techmap data
│   ├── 02_eda_v2.ipynb              EDA on the v2 fraud dataset
│   └── build_eda_v2.py              regenerates 02_eda_v2.ipynb
├── reports/
│   ├── eda_v2.pdf                   full EDA report with dataset card (Checkpoint 2)
│   ├── baseline_results.csv         all baseline metrics
│   ├── figures/checkpoint1/         Checkpoint 1 charts
│   ├── figures/v2/                  Checkpoint 2 charts
│   └── v1_eda/                      charts from the first pipeline
├── docs/
│   └── dataset_card.md
└── JobAI/                     resume parser app prototype (Phase 1), see JobAI/README.md
```

## Data

| Dataset | Use | Source |
|---|---|---|
| Fake job postings (`fake_job_postings.csv`) | Labeled real/fraud postings, the fraud target | Kaggle |
| LinkedIn Job Postings 2023-24 | Extra real postings, Checkpoint 1 EDA | Kaggle: arshkon/linkedin-job-postings |
| US Job Postings 2023-05-05 | Extra real postings, Checkpoint 1 EDA | Kaggle: techmap/us-job-postings-from-2023-05-05 |
| Synthetic `Fake_Postings.csv` | Not used for training (one repeated template); comparison only | Kaggle |

Raw files are not stored in git. See `data/README.md`.

## Run the pipeline

```
conda create -n jobai python=3.11 && conda activate jobai
pip install -r requirements.txt

# 1. put raw files in data/raw/ (see data/README.md), then:
python src/data/flatten_techmap.py --zip data/raw/us_jobpostings_kaggle.zip --out data/raw
python src/data/build_dataset.py --with-synthetic     # about 2 min
python src/models/train_baseline.py                   # about 2.5 min

# 2. upload logged runs to Weights & Biases
wandb login
wandb sync reports/wandb/offline-run-*
```

## Current results (Checkpoint 2)

v2 dataset: 20,288 postings, 681 fraud (3.4%), split 70/15/15 by company.
Scores below are on the real labeled test postings.

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Always "real" (96% accuracy) | 0 | 0 | 0 | 0.04 |
| Rule: no company or has email | 0.12 | 0.64 | 0.20 | 0.09 |
| TF-IDF + logistic regression (cutoff 0.73) | 0.87 | 0.73 | 0.79 | 0.83 |
| Same model trained with synthetic data | 0.89 | 0.50 | 0.64 | 0.81 |

Details: `reports/baseline_results.csv`, `reports/eda_v2.pdf`.

## Progress

| Checkpoint | Done |
|---|---|
| 1 | Project plan, tool setup, EDA on LinkedIn + Techmap (`notebooks/01_eda_checkpoint1.ipynb`) |
| 2 | ML-ready fraud dataset, EDA PDF + dataset card, baseline model with W&B tracking |

## Team workflow

- Work on a branch, open a pull request, have one teammate review.
- Keep data and model files out of git (`.gitignore` handles this); share them on Google Drive.
- Write Google-style docstrings for every function.
