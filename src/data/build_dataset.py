"""Build the v2 ML-ready dataset for the job posting fraud classifier.

Combines labeled real/fraud postings (ds1) with extra real postings from the
LinkedIn and Techmap datasets, cleans and harmonizes the fields, removes
duplicates, adds engineered features, and writes stratified, company-grouped
train/val/test splits.

Sources: ds1 = Kaggle fake_job_postings.csv (17,880 postings, 866 fraud), plus a
fixed number of extra real postings from LinkedIn and Techmap.

The synthetic "Fake Postings" data (Fake_Postings.csv) is NOT used for the main dataset. With
``--with-synthetic`` an extra ``train_with_synthetic.csv`` is written (same
test set) so its effect can be measured.

Example:
    python src/data/build_dataset.py --raw data/raw --out data/processed/v2
"""

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
MIN_WORDS = 10  # drop postings whose description is basically empty

URL_RE = re.compile(r"(#URL_[0-9a-f]+#|https?://\S+|www\.\S+)", re.I)
EMAIL_RE = re.compile(r"(#EMAIL_[0-9a-f]+#|[\w.+-]+@[\w-]+\.[\w.]+)", re.I)
PHONE_RE = re.compile(r"(#PHONE_[0-9a-f]+#|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4})", re.I)
TAG_RE = re.compile(r"<[^>]+>")
MONEY_RE = re.compile(r"(?:\$\s?\d|earn\b|per week|/week|weekly pay|cash)", re.I)


def clean_text(text):
    """Normalize posting text so all sources look alike.

    Unescapes HTML, strips tags, replaces URLs/emails/phones (real ones and the
    #URL_..# style placeholders) with the same plain tokens, splits words that
    were glued together (``programSupport`` -> ``program Support``) and
    collapses whitespace.

    Args:
        text: Raw description or title (may be NaN).

    Returns:
        Cleaned string ("" for missing input).
    """
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = TAG_RE.sub(" ", t)
    t = URL_RE.sub(" urltoken ", t)
    t = EMAIL_RE.sub(" emailtoken ", t)
    t = PHONE_RE.sub(" phonetoken ", t)
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    return re.sub(r"\s+", " ", t).strip()


def norm_employment(value):
    """Map the many employment-type spellings to one small label set."""
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    v = value.lower()
    if "full" in v and "part" in v:
        return "full_or_part_time"
    if "full" in v or "permanent" in v:
        return "full_time"
    if "part" in v:
        return "part_time"
    if "contract" in v or "temp" in v:
        return "contract_temp"
    if "intern" in v:
        return "internship"
    return "other"


def is_present(series):
    """1 if a field has a real (non-empty) value, else 0."""
    s = series.astype("string").str.strip()
    return (s.notna() & (s != "") & (s.str.lower() != "nan")).astype(int)


def company_key(name, fallback_ids):
    """Group key for splitting: same company -> same split.

    Uses an md5 of the lowercased company text; rows without a company get
    their own unique key so they can land in any split.
    """
    keys = []
    for n, fid in zip(name, fallback_ids):
        if isinstance(n, str) and n.strip():
            keys.append(hashlib.md5(n.strip().lower().encode()).hexdigest()[:12])
        else:
            keys.append(f"nocompany_{fid}")
    return keys


# ---------------------------------------------------------------- loaders

def load_ds1(raw):
    """Load the labeled real/fraud postings (ds1).

    Prefers the original Kaggle file ``fake_job_postings.csv`` (EMSCAD format).
    Falls back to the ds1 rows inside Doron's ``balanced_all.csv``, which only
    has a sample of ds1's real postings.
    """
    full = raw / "fake_job_postings.csv"
    if full.exists():
        d = pd.read_csv(full)
        desc = (d["description"].fillna("") + " " + d["requirements"].fillna("")
                + " " + d["benefits"].fillna(""))
        out = pd.DataFrame({
            "title": d["title"], "company_text": d["company_profile"],
            "location": d["location"], "employment_type": d["employment_type"],
            "experience": d["required_experience"], "salary_raw": d["salary_range"],
            "description": desc, "fraudulent": d["fraudulent"].astype(int)})
        note = "full Kaggle file"
    else:
        d = pd.read_csv(raw / "doron_balanced_all.csv", low_memory=False)
        d = d[d["source"] == "ds1_srisai"]
        out = pd.DataFrame({
            "title": d["title"], "company_text": d["company"],
            "location": d["location"], "employment_type": d["employment_type"],
            "experience": d["experience"], "salary_raw": d["salary"],
            "description": d["description"], "fraudulent": d["fraudulent"].astype(int)})
        note = "sample from Doron's balanced_all.csv"
    out["source"] = "ds1_labeled"
    out["label_source"] = "labeled"
    print(f"ds1 ({note}): {len(out):,} rows, fraud={out.fraudulent.sum():,}")
    return out


def load_linkedin(raw):
    """Load LinkedIn postings (all assumed real)."""
    d = pd.read_csv(raw / "linkedin" / "postings.csv",
                    usecols=["title", "company_name", "location", "formatted_work_type",
                             "formatted_experience_level", "normalized_salary", "description"])
    return pd.DataFrame({
        "title": d["title"], "company_text": d["company_name"], "location": d["location"],
        "employment_type": d["formatted_work_type"],
        "experience": d["formatted_experience_level"],
        "salary_raw": d["normalized_salary"], "description": d["description"],
        "fraudulent": 0, "source": "linkedin", "label_source": "assumed_real"})


def load_techmap(raw):
    """Load flattened Techmap postings (all assumed real).

    Uses the Checkpoint 1 flattener output, which keeps company, city/state,
    work type, career level and salary text from the nested JSON.
    """
    d = pd.read_csv(raw / "techmap_flat.csv", low_memory=False)
    text = pd.read_json(raw / "techmap_text.jsonl", lines=True)[["id", "text"]]
    d = d.merge(text, on="id", how="left")
    loc = (d["city"].fillna("") + ", " + d["state"].fillna("")).str.strip(", ")
    return pd.DataFrame({
        "title": d["title"], "company_text": d["company"], "location": loc.replace("", np.nan),
        "employment_type": d["workType"], "experience": d["careerLevel"],
        "salary_raw": d["salary_text"], "description": d["text"],
        "fraudulent": 0, "source": "techmap", "label_source": "assumed_real"})


def load_synthetic(raw):
    """Load the synthetic 'Fake Postings' rows (only used for the comparison)."""
    d = pd.read_csv(raw / "Fake_Postings.csv")
    desc = (d["description"].fillna("") + " " + d["requirements"].fillna("")
            + " " + d["benefits"].fillna(""))
    return pd.DataFrame({
        "title": d["title"], "company_text": d["company_profile"], "location": d["location"],
        "employment_type": d["employment_type"], "experience": np.nan,
        "salary_raw": d["salary_range"], "description": desc,
        "fraudulent": 1, "source": "synthetic_fake_postings", "label_source": "labeled"})


# ---------------------------------------------------------------- features

def add_features(df):
    """Clean text fields and add engineered features used by the models."""
    df = df.copy()
    df["title"] = df["title"].map(clean_text)
    df["description"] = df["description"].map(clean_text)
    df["employment_type"] = df["employment_type"].map(norm_employment)
    df["has_company"] = is_present(df["company_text"])
    df["has_location"] = is_present(df["location"])
    df["has_salary"] = is_present(df["salary_raw"])
    df["has_experience"] = is_present(df["experience"])
    df["has_employment_type"] = (df["employment_type"] != "unknown").astype(int)
    df["desc_words"] = df["description"].str.split().str.len()
    df["title_words"] = df["title"].str.split().str.len()
    df["has_url"] = df["description"].str.contains("urltoken").astype(int)
    df["has_email"] = df["description"].str.contains("emailtoken").astype(int)
    df["has_phone"] = df["description"].str.contains("phonetoken").astype(int)
    df["money_words"] = df["description"].str.contains(MONEY_RE).astype(int)
    letters = df["title"].str.replace(r"[^A-Za-z]", "", regex=True)
    df["title_all_caps"] = ((letters.str.len() > 3) & letters.str.isupper()).astype(int)
    return df


def dedupe(df):
    """Drop exact duplicates of title + description (keeping a fraud copy if labels clash)."""
    key = (df["title"].str.lower() + "|" + df["description"].str.lower()).map(
        lambda s: hashlib.md5(s.encode()).hexdigest())
    df = df.assign(_key=key).sort_values("fraudulent", ascending=False)
    out = df.drop_duplicates("_key").drop(columns="_key")
    return out, len(df) - len(out)


def split(df, seed=SEED):
    """70/15/15 split, stratified by label and grouped by company."""
    y, g = df["fraudulent"].values, df["company_key"].values
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(outer.split(df, y, g))
    rest = df.iloc[rest_idx]
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(inner.split(rest, rest["fraudulent"], rest["company_key"]))
    return rest.iloc[tr_idx], rest.iloc[va_idx], df.iloc[test_idx]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="data/raw", type=Path)
    ap.add_argument("--out", default="data/processed/v2", type=Path)
    ap.add_argument("--extra-real", type=int, default=4500,
                    help="extra real postings to add from LinkedIn (2/3) and Techmap (1/3); 0 = ds1 only")
    ap.add_argument("--with-synthetic", action="store_true",
                    help="also write train_with_synthetic.csv for the comparison run")
    args = ap.parse_args()
    log = {}

    # 1. load + clean each source separately so we can dedupe within it
    ds1 = add_features(load_ds1(args.raw))
    # LinkedIn has 124k rows; clean a random pool of 20k (far more than we sample)
    li = add_features(load_linkedin(args.raw).sample(20000, random_state=SEED))
    tm = add_features(load_techmap(args.raw))

    # 2. drop near-empty descriptions and duplicates
    parts = []
    for name, d in [("ds1_labeled", ds1), ("linkedin", li), ("techmap", tm)]:
        n0 = len(d)
        d = d[d["desc_words"] >= MIN_WORDS]
        d, ndup = dedupe(d)
        log[name] = {"raw": n0, "short_dropped": n0 - len(d) - ndup, "dupes_dropped": ndup,
                     "kept": len(d)}
        parts.append(d)
    ds1, li, tm = parts

    # 3. add a fixed number of extra real postings (2/3 LinkedIn, 1/3 Techmap) so the
    #    model also sees recent postings from other job boards, not only ds1's
    need = args.extra_real
    n_li, n_tm = int(round(need * 2 / 3)), need - int(round(need * 2 / 3))
    li = li.sample(min(n_li, len(li)), random_state=SEED)
    tm = tm.sample(min(n_tm, len(tm)), random_state=SEED)

    df = pd.concat([ds1, li, tm], ignore_index=True)
    df, cross_dupes = dedupe(df)  # same text across sources
    log["cross_source_dupes_dropped"] = cross_dupes
    df["posting_id"] = [f"p{i:06d}" for i in range(len(df))]
    df["company_key"] = company_key(df["company_text"], df["posting_id"])

    # 4. split
    train, val, test = split(df)
    keep = ["posting_id", "source", "label_source", "fraudulent", "title", "description",
            "employment_type", "has_company", "has_location", "has_salary", "has_experience",
            "has_employment_type", "desc_words", "title_words", "has_url", "has_email",
            "has_phone", "money_words", "title_all_caps", "company_key"]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[keep].to_csv(args.out / f"{name}.csv", index=False)
        log[f"{name}_counts"] = {f"{src}|fraud={lab}": int(n) for (src, lab), n
                                 in part.groupby(["source", "fraudulent"]).size().items()}
        log[f"{name}_rows"] = len(part)
    # sanity: no company in two splits
    log["company_overlap_train_test"] = int(len(set(train.company_key) & set(test.company_key)))

    # 5. optional comparison set: train + synthetic rows (test set unchanged)
    if args.with_synthetic:
        syn = add_features(load_synthetic(args.raw))
        syn["posting_id"] = [f"s{i:06d}" for i in range(len(syn))]
        syn["company_key"] = company_key(syn["company_text"], syn["posting_id"])
        pd.concat([train, syn])[keep].to_csv(args.out / "train_with_synthetic.csv", index=False)
        log["synthetic_rows_added"] = len(syn)

    (args.out / "build_log.json").write_text(json.dumps(log, indent=2, default=int))
    print(json.dumps(log, indent=2, default=int))


if __name__ == "__main__":
    main()
