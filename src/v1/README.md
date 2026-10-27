# v1 pipeline (Doron)

First version of the merged fraud dataset. Kept for reference; replaced by `src/data/build_dataset.py`.

| Script | What it does |
|---|---|
| `json_to_csv.py` | Converts the Techmap JSON to CSV |
| `merge_balance_jobs.py` | Merges 4 sources, dedupes, balances real:fraud 50:50, splits 70/15/15 |
| `eda_and_cleaning.py` | EDA charts (now in `reports/v1_eda/`) and cleaned splits (now in `data/v1/`) |
| `dataframes.py` | Quick shape check of the source files |

Why v2 replaced it: 94% of the fraud rows came from the synthetic `Fake_Postings.csv`, where every
description is the same short template. A one-line rule ("description under 20 words") scored 96%
accuracy on the v1 test set, so models could not learn real fraud patterns. v2 uses the full labeled
`fake_job_postings.csv` instead. Details in `reports/eda_v2.pdf`, section 10.
