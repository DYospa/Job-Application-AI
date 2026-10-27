# Data

Raw and processed data are not stored in git (too large). Put files here locally.

## Raw files (`data/raw/`)

| File | Download from |
|---|---|
| `fake_job_postings.csv` | Kaggle fake job postings dataset (17,880 postings, 866 fraud) |
| `Fake_Postings.csv` | Kaggle synthetic fake postings (comparison only) |
| `linkedin/postings.csv` | https://www.kaggle.com/datasets/arshkon/linkedin-job-postings (unzip into `linkedin/`) |
| `us_jobpostings_kaggle.zip` | https://www.kaggle.com/datasets/techmap/us-job-postings-from-2023-05-05 |

Then run `python src/data/flatten_techmap.py --zip data/raw/us_jobpostings_kaggle.zip --out data/raw`
to create `techmap_flat.csv` and `techmap_text.jsonl`.

## Processed data (`data/processed/v2/`)

Created by `python src/data/build_dataset.py`. Shared copy: **[add Google Drive link]**

| File | Rows | Fraud |
|---|---|---|
| train.csv | 14,492 | 487 |
| val.csv | 2,898 | 97 |
| test.csv | 2,898 | 97 |
| train_with_synthetic.csv | train + 10,000 synthetic | comparison only |
| build_log.json | counts at each build step | |

## v1 (`data/v1/`)

First merged dataset from `src/v1/` (four sources, balanced 50/50). Kept for reference. Not used
for training: 94% of its fraud rows came from the synthetic file. See `reports/eda_v2.pdf`, section 10.
