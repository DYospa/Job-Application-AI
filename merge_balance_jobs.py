"""
Merge four Kaggle job-posting datasets, dedupe, balance real:fraud 50:50,
and write stratified train/val/test splits.

Expected downloads (edit PATHS below):
  ds1: srisaisuhassanisetty/fake-job-postings      -> fake_job_postings.csv
  ds2: shivamb/real-or-fake-fake-jobposting-prediction -> fake_job_postings.csv
  ds3: techmap/us-job-postings-from-2023-05-05     -> *.csv (large)
  ds4: arshkon/linkedin-job-postings               -> postings.csv / job_postings.csv

Usage:  pip install pandas scikit-learn
        python merge_balance_jobs.py
"""
import hashlib
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------- config ----------------
BASE = "/Users/doronyospa/Desktop/JobAI/data"
PATHS = {
    "ds1": f"{BASE}/fake_job_postings.csv",
    "ds2": f"{BASE}/Fake Postings.csv",
    "ds3": f"{BASE}/techmap-jobs_us_2023-05-05.csv",   # check actual filename
    "ds4": f"{BASE}/merged_linkedin_jobs.csv",
}
TECHMAP_SAMPLE = 200_000   # rows to read from the huge Techmap file (all assumed real)
RANDOM_STATE = 42
SPLITS = (0.70, 0.15, 0.15)  # train / val / test
OUT_DIR = Path("output")

COMMON = ["title", "company", "location", "description", "employment_type",
          "experience", "salary", "fraudulent", "source", "label_source"]


def first_col(df, candidates):
    """Return the first candidate column present in df, else None."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def harmonize(df, source, fraud_labeled):
    mapping = {
        "title": ["title", "job_title", "name"],
        "company": ["company_name", "company", "company_profile"],
        "location": ["location", "city", "state"],
        "description": ["description", "text", "job_description"],
        "employment_type": ["employment_type", "work_type", "type"],
        "experience": ["required_experience", "formatted_experience_level", "experience_level"],
        "salary": ["salary_range", "normalized_salary", "med_salary", "salary"],
    }
    out = pd.DataFrame(index=df.index)
    for target, cands in mapping.items():
        col = first_col(df, cands)
        out[target] = df[col] if col else pd.NA
    if fraud_labeled:
        out["fraudulent"] = df["fraudulent"].astype(int)
        out["label_source"] = "labeled"
    else:
        out["fraudulent"] = 0
        out["label_source"] = "assumed_real"
    out["source"] = source
    return out[COMMON]


def dedupe_key(row):
    text = f"{row['title']}|{row['description']}"
    text = re.sub(r"\W+", " ", str(text).lower()).strip()
    return hashlib.md5(text.encode()).hexdigest()


# ---------------- load ----------------
frames = []
frames.append(harmonize(pd.read_csv(PATHS["ds1"]), "ds1_srisai", True))
frames.append(harmonize(pd.read_csv(PATHS["ds2"]), "ds2_shivamb", True))
frames.append(harmonize(pd.read_csv(PATHS["ds3"], nrows=TECHMAP_SAMPLE,
                                    on_bad_lines="skip", low_memory=False),
                        "ds3_techmap", False))
frames.append(harmonize(pd.read_csv(PATHS["ds4"]), "ds4_linkedin", False))

df = pd.concat(frames, ignore_index=True)
print("Raw rows:", len(df))

# ---------------- clean + dedupe ----------------
df = df.dropna(subset=["description"])
df = df[df["description"].astype(str).str.len() > 50]
df["dedupe_key"] = df.apply(dedupe_key, axis=1)

# If a duplicate appears with conflicting labels, keep the fraud=1 version
df = (df.sort_values("fraudulent", ascending=False)
        .drop_duplicates("dedupe_key", keep="first")
        .drop(columns="dedupe_key")
        .reset_index(drop=True))
print("After dedupe:", len(df))
print(df.groupby(["source", "fraudulent"]).size())

# ---------------- balance 50:50 ----------------
fraud = df[df["fraudulent"] == 1]
real = df[df["fraudulent"] == 0]
n = min(len(fraud), len(real))
balanced = pd.concat([
    fraud.sample(n, random_state=RANDOM_STATE),
    real.sample(n, random_state=RANDOM_STATE),
]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Balanced: {n} fraud + {n} real = {len(balanced)}")

# ---------------- stratified split ----------------
train_f, val_f, test_f = SPLITS
train, rest = train_test_split(balanced, train_size=train_f,
                               stratify=balanced["fraudulent"],
                               random_state=RANDOM_STATE)
val, test = train_test_split(rest, train_size=val_f / (val_f + test_f),
                             stratify=rest["fraudulent"],
                             random_state=RANDOM_STATE)

OUT_DIR.mkdir(exist_ok=True)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(OUT_DIR / f"{name}.csv", index=False)
    print(name, len(part), part["fraudulent"].value_counts().to_dict())
balanced.to_csv(OUT_DIR / "balanced_all.csv", index=False)
