"""
EDA + cleaning for the balanced fake-job-postings dataset.

Input : output/balanced_all.csv   (from merge_balance_jobs.py)
Output: output/eda/*.png, *.csv   (EDA artifacts)
        output/clean/train.csv, val.csv, test.csv, clean_all.csv

Usage:  pip install pandas scikit-learn matplotlib
        python eda_and_cleaning.py
"""
import html
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------- config ----------------
IN_PATH = Path("output/balanced_all.csv")
EDA_DIR = Path("output/eda")
CLEAN_DIR = Path("output/clean")
RANDOM_STATE = 42
SPLITS = (0.70, 0.15, 0.15)
MIN_DESC_WORDS = 10
# Set True to drop the unlabeled sources (ds3/ds4) so "real" only comes from the
# labeled datasets. This removes the source-leakage risk but shrinks the data.
LABELED_ONLY = False

EDA_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

STOPWORDS = set("""a an the and or of to in for on with at by from as is are was were be been this that
these those it its our we you your they their will can have has had not but if all any more such than
into out up about who what which when where how also may must should would could""".split())


def savefig(name):
    plt.tight_layout()
    plt.savefig(EDA_DIR / name, dpi=130)
    plt.close()


# ====================================================
# 1. LOAD
# ====================================================
df = pd.read_csv(IN_PATH, low_memory=False)
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nClass balance:\n", df["fraudulent"].value_counts())

if LABELED_ONLY:
    df = df[df["label_source"] == "labeled"].copy()
    print("LABELED_ONLY on -> shape:", df.shape)

# ====================================================
# 2. RAW EDA
# ====================================================
# 2a. Missingness by class
missing = (df.isna().groupby(df["fraudulent"]).mean().T * 100).round(1)
missing.columns = ["real_%missing", "fraud_%missing"]
print("\nMissing % by class:\n", missing)
missing.to_csv(EDA_DIR / "missingness_by_class.csv")

# 2b. SOURCE LEAKAGE CHECK (important!)
# If fraud comes only from ds1/ds2 but "real" also comes from ds3/ds4, a model can
# learn "which dataset is this" instead of "is this fraud".
src = pd.crosstab(df["source"], df["fraudulent"])
print("\nSource x class (leakage check):\n", src)
src.to_csv(EDA_DIR / "source_by_class.csv")
src.plot(kind="bar", stacked=True, title="Source vs class")
savefig("source_by_class.png")
if (src.get(1, pd.Series(dtype=int)) > 0).sum() < (src.get(0, pd.Series(dtype=int)) > 0).sum():
    print("\n WARNING: real postings come from sources that contain no fraud examples.\n"
          "   Consider LABELED_ONLY = True, or evaluate separately on ds1/ds2 only.")

# 2c. Text length features
df["desc_words"] = df["description"].fillna("").str.split().str.len()
df["title_words"] = df["title"].fillna("").str.split().str.len()
print("\nDescription length by class:\n", df.groupby("fraudulent")["desc_words"].describe())

for col in ["desc_words", "title_words"]:
    cap = df[col].quantile(0.99)
    for label, name in [(0, "real"), (1, "fraud")]:
        df.loc[df["fraudulent"] == label, col].clip(upper=cap).plot(
            kind="hist", bins=50, alpha=0.5, label=name)
    plt.legend(); plt.title(col); plt.xlabel(col)
    savefig(f"hist_{col}.png")

# 2d. Categorical distributions by class
for col in ["employment_type", "experience"]:
    top = df[col].fillna("MISSING").str.lower().value_counts().head(8).index
    tab = pd.crosstab(df[col].fillna("MISSING").str.lower(), df["fraudulent"], normalize="columns")
    tab.loc[tab.index.isin(top)].plot(kind="bar", title=f"{col} (share within class)")
    savefig(f"cat_{col}.png")

# 2e. Suspicious-pattern rates (quick signal check)
patterns = {
    "has_email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "has_url": r"https?://|www\.|#URL_",
    "has_phone": r"\+?\d[\d\s().-]{8,}\d",
    "money_words": r"\$\s?\d|per (hour|week)|weekly pay|earn",
    "urgency": r"urgent|immediately|asap|no experience|work from home",
    "all_caps_title": None,
}
rates = {}
desc = df["description"].fillna("")
for k, pat in patterns.items():
    if pat:
        df[k] = desc.str.contains(pat, case=False, regex=True)
    else:
        df[k] = df["title"].fillna("").map(lambda t: t.isupper() and len(t) > 3)
    rates[k] = df.groupby("fraudulent")[k].mean().round(3)
rates = pd.DataFrame(rates).T.rename(columns={0: "real", 1: "fraud"})
print("\nPattern rates by class:\n", rates)
rates.to_csv(EDA_DIR / "pattern_rates.csv")
rates.plot(kind="bar", title="Suspicious-pattern rate by class")
savefig("pattern_rates.png")

# 2f. Top words by class
def top_words(texts, n=25):
    c = Counter()
    for t in texts:
        c.update(w for w in re.findall(r"[a-z]{3,}", str(t).lower()) if w not in STOPWORDS)
    return c.most_common(n)

tw = {name: top_words(df.loc[df["fraudulent"] == lab, "description"].fillna("").sample(
          min(5000, (df["fraudulent"] == lab).sum()), random_state=RANDOM_STATE))
      for lab, name in [(0, "real"), (1, "fraud")]}
pd.DataFrame({k: [f"{w} ({c})" for w, c in v] for k, v in tw.items()}).to_csv(
    EDA_DIR / "top_words_by_class.csv", index=False)

df = df.drop(columns=list(patterns.keys()) + ["desc_words", "title_words"])

# ====================================================
# 3. CLEANING
# ====================================================
def clean_text(s):
    if pd.isna(s):
        return ""
    s = html.unescape(str(s))
    s = re.sub(r"<[^>]+>", " ", s)                              # strip HTML tags
    s = re.sub(r"#(URL|EMAIL|PHONE)_[0-9a-f]+#", r" <\1> ", s)  # EMSCAD anonymization tokens
    s = re.sub(r"https?://\S+|www\.\S+", " <URL> ", s)
    s = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", " <EMAIL> ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_cat(s):
    if pd.isna(s):
        return "unknown"
    s = re.sub(r"[^a-z]+", "_", str(s).lower()).strip("_")
    return s or "unknown"


clean = df.copy()

# keep missingness signals BEFORE filling (they can be predictive)
for col in ["company", "location", "salary", "employment_type", "experience"]:
    clean[f"has_{col}"] = clean[col].notna().astype(int)

for col in ["title", "company", "location", "description"]:
    clean[col] = clean[col].map(clean_text)

# salary: keep as string flag only (formats differ wildly across sources)
clean = clean.drop(columns=["salary"])

for col in ["employment_type", "experience"]:
    clean[col] = clean[col].map(norm_cat)

# drop empty / tiny descriptions
n0 = len(clean)
clean = clean[clean["description"].str.split().str.len() >= MIN_DESC_WORDS]
print(f"\nDropped {n0 - len(clean)} rows with < {MIN_DESC_WORDS}-word descriptions")

# dedupe again on cleaned text (cleaning can expose new duplicates)
clean["_key"] = (clean["title"].str.lower() + "|" + clean["description"].str.lower())
n0 = len(clean)
clean = (clean.sort_values("fraudulent", ascending=False)
              .drop_duplicates("_key").drop(columns="_key"))
print(f"Dropped {n0 - len(clean)} duplicates after cleaning")

# engineered features
clean["desc_len_words"] = clean["description"].str.split().str.len()
clean["title_len_words"] = clean["title"].str.split().str.len()
clean["has_url_token"] = clean["description"].str.contains("<URL>").astype(int)
clean["has_email_token"] = clean["description"].str.contains("<EMAIL>").astype(int)
clean["title_all_caps"] = clean["title"].map(lambda t: int(t.isupper() and len(t) > 3))
clean["text"] = (clean["title"] + ". " + clean["description"]).str.strip()

# ====================================================
# 4. RE-BALANCE + SPLIT
# ====================================================
fraud, real = clean[clean["fraudulent"] == 1], clean[clean["fraudulent"] == 0]
n = min(len(fraud), len(real))
bal = pd.concat([fraud.sample(n, random_state=RANDOM_STATE),
                 real.sample(n, random_state=RANDOM_STATE)]
                ).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"\nRe-balanced after cleaning: {n} fraud + {n} real")

tr, va, te = SPLITS
train, rest = train_test_split(bal, train_size=tr, stratify=bal["fraudulent"], random_state=RANDOM_STATE)
val, test = train_test_split(rest, train_size=va / (va + te), stratify=rest["fraudulent"], random_state=RANDOM_STATE)

bal.to_csv(CLEAN_DIR / "clean_all.csv", index=False)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(CLEAN_DIR / f"{name}.csv", index=False)
    print(f"{name}: {len(part)} rows, class counts {part['fraudulent'].value_counts().to_dict()}")

print(f"\nDone. EDA files in {EDA_DIR}/, cleaned data in {CLEAN_DIR}/")
