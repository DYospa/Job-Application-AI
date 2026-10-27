# Dataset card: JobAI fraud postings v2 (draft)

## Summary
Job postings labeled real (0) or fraudulent (1), built to train the JobAI fraud
filter. 20,288 postings, 681 fraudulent (3.4%). Split 70/15/15 into train/val/test.

## Sources
| Source | Rows used | Label | Notes |
|---|---|---|---|
| ds1: Kaggle fake_job_postings.csv (17,880 rows, 866 fraud) | 15,107 real, 681 fraud after cleaning | labeled | Only labeled fraud source. 2,075 duplicates removed |
| LinkedIn Job Postings 2023-24 (arshkon, Kaggle) | 3,000 | assumed real | Random sample from a 20k pool |
| US Job Postings 2023-05-05 (techmap, Kaggle) | 1,500 | assumed real | Random sample, flattened from nested JSON |
| Synthetic Fake_Postings.csv (10,000 rows) | 0 (excluded) | labeled fraud | Generated text with one repeated template; only used in a comparison run |

## Lineage (src/data/build_dataset.py)
1. Load each source (ds1 description = description + requirements + benefits) and map it to one schema: title, description, company, location,
   employment type, experience, salary.
2. Clean text: unescape HTML, strip tags, split glued words, replace URLs, emails and
   phone numbers with `urltoken`, `emailtoken`, `phonetoken` in every source.
3. Drop descriptions under 10 words; drop exact duplicate title + description.
4. Add 3,000 LinkedIn and 1,500 Techmap real postings (random sample, seed 42).
5. Engineer features (below).
6. Split 70/15/15 with StratifiedGroupKFold: fraud share equal in each split, no company
   appears in more than one split. Seed 42.

## Fields
| Field | Type | Meaning |
|---|---|---|
| posting_id | str | Row id |
| source | str | ds1_labeled, linkedin, techmap |
| label_source | str | labeled or assumed_real |
| fraudulent | int | Target: 1 = fraud |
| title, description | str | Cleaned text |
| employment_type | str | full_time, part_time, contract_temp, internship, full_or_part_time, other, unknown |
| has_company, has_location, has_salary, has_experience, has_employment_type | 0/1 | Field is filled in |
| has_url, has_email, has_phone | 0/1 | Description contains one |
| money_words | 0/1 | Mentions $ amounts, "earn", "per week", "cash" |
| title_all_caps | 0/1 | Title written in all caps |
| desc_words, title_words | int | Word counts |
| company_key | str | Hashed company, used for grouped splitting |

## Known issues and limitations
- Small fraud set: 681 examples, 97 in test, so metrics have wide error bars.
- LinkedIn and Techmap labels are assumed real; a few scams may be in there.
- Source differences: ds1 descriptions are shorter than LinkedIn/Techmap, and ds1 "company"
  is a company profile, not a name. Report metrics on ds1-only test to avoid rewarding source detection.
- Fraud examples come from one older dataset, so new scam styles may not be covered.
- English, mostly U.S. postings.

## Intended use
Training and evaluating the JobAI fraud warning. Not for judging specific employers.

## License
Follows the Kaggle licenses of each source (check each dataset page before sharing).
