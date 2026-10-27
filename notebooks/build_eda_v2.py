"""Builds notebooks/02_eda_v2.ipynb (run: python notebooks/build_eda_v2.py)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# JobAI: EDA on the ML-ready fraud dataset (v2)
Checkpoint 2. Target: `fraudulent` (1 = fake posting). Dataset built by `src/data/build_dataset.py`.
""")

code(r'''
# setup
import json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif

DATA = "../data/processed/v2"
RAW = "../data/raw"
FIG = "../reports/figures/v2"

# one color per class, used on every chart
REAL, FRAUD, SYN, GRAY = "#2a78d6", "#eb6834", "#6250d6", "#9a998f"
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
def save(name):
    plt.savefig(f"{FIG}/{name}", facecolor="white"); plt.show()

# load all three splits into one frame (split column kept)
df = pd.concat([pd.read_csv(f"{DATA}/{s}.csv").assign(split=s) for s in ("train", "val", "test")],
               ignore_index=True)
df["label"] = df["fraudulent"].map({0: "real", 1: "fraud"})
ds1 = df[df["source"] == "ds1_labeled"]          # the part with real labels for both classes
build_log = json.load(open(f"{DATA}/build_log.json"))
print(f"{len(df):,} postings x {df.shape[1]} columns")

# metadata: type, % missing and an example value for every column
meta = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing_%": (df.isna().mean() * 100).round(1),
                     "unique": df.nunique(),
                     "example": [str(df[c].dropna().iloc[0])[:40] for c in df.columns]})
meta
''')

md("## 1. Lineage: how the dataset was built")
code(r'''
# rows kept at each step, per source (from the build log)
steps = pd.DataFrame({k: build_log[k] for k in ("ds1_labeled", "linkedin", "techmap")}).T
steps["used_in_v2"] = [int((df.source == s).sum()) for s in steps.index]
steps
''')
code(r'''
# funnel chart for the labeled source
lab = build_log["ds1_labeled"]
stages = ["raw file", "after dropping\nnear-empty posts", "after removing\nduplicates"]
vals = [lab["raw"], lab["raw"] - lab["short_dropped"], lab["kept"]]
fig, ax = plt.subplots(figsize=(6.5, 2.6))
ax.barh(stages[::-1], vals[::-1], color=REAL, height=0.55)
for i, v in enumerate(vals[::-1]):
    ax.text(v + 150, i, f"{v:,}", va="center", fontsize=9)
ax.set_xlim(0, max(vals) * 1.15); ax.grid(axis="y", visible=False)
ax.set_title("ds1 (fake_job_postings.csv): rows kept at each cleaning step")
save("v2_lineage_ds1.png")
''')

md("## 2. Class balance and sources")
code(r'''
# rows and fraud rate per split
bal = df.groupby("split").agg(rows=("fraudulent", "size"), fraud=("fraudulent", "sum"))
bal["fraud_%"] = (bal.fraud / bal.rows * 100).round(2)
bal.loc[["train", "val", "test"]]
''')
code(r'''
# where each class comes from
src = pd.crosstab(df.source, df.label)[["real", "fraud"]]
fig, ax = plt.subplots(figsize=(6.5, 2.8))
left = np.zeros(len(src))
for col, color in (("real", REAL), ("fraud", FRAUD)):
    ax.barh(src.index, src[col], left=left, color=color, height=0.55, label=col)
    left += src[col].values
for i, (r, f) in enumerate(src.values):
    ax.text(r + f + 150, i, f"{r:,} real / {f:,} fraud", va="center", fontsize=8.5)
ax.set_xlim(0, src.values.sum(axis=1).max() * 1.45); ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="upper right"); ax.set_title("Postings by source and class")
save("v2_source_by_class.png")
print(f"overall fraud rate: {df.fraudulent.mean():.1%}")
''')

md("## 3. Missing fields by class (feature vs target)")
code(r'''
# share of postings with each field filled in, real vs fraud (ds1 only, so both classes come from the same source)
fields = ["has_company", "has_location", "has_salary", "has_experience", "has_employment_type"]
rates = ds1.groupby("label")[fields].mean().T[["real", "fraud"]] * 100
fig, ax = plt.subplots(figsize=(6.5, 3.2))
y = np.arange(len(fields))
ax.barh(y + 0.2, rates.real, height=0.38, color=REAL, label="real")
ax.barh(y - 0.2, rates.fraud, height=0.38, color=FRAUD, label="fraud")
ax.set_yticks(y, [f.replace("has_", "") for f in fields]); ax.invert_yaxis()
ax.xaxis.set_major_formatter(mtick.PercentFormatter()); ax.set_xlim(0, 105)
ax.grid(axis="y", visible=False); ax.legend(frameon=False, loc="lower right")
ax.set_title("% of postings with the field filled in (ds1)")
save("v2_fields_by_class.png")
rates.round(1)
''')

md("## 4. Description length by class")
code(r'''
# word count distribution, log scale because lengths are skewed (ds1 only)
fig, ax = plt.subplots(figsize=(6.5, 3.2))
bins = np.logspace(1, np.log10(3000), 40)
for lab, color in (("real", REAL), ("fraud", FRAUD)):
    v = ds1.loc[ds1.label == lab, "desc_words"]
    ax.hist(v, bins=bins, density=True, alpha=0.6, color=color, label=f"{lab} (median {v.median():.0f})")
ax.set_xscale("log"); ax.set_yticks([]); ax.set_xlabel("Words in description (log scale)")
ax.legend(frameon=False); ax.set_title("Description length, real vs fraud (ds1)")
save("v2_desc_length_by_class.png")
ds1.groupby("label").desc_words.describe().round(0)
''')

md("## 5. Text patterns by class")
code(r'''
# share of postings containing a URL, email, phone, money words, or an all-caps title (ds1 only)
pats = ["has_url", "has_email", "has_phone", "money_words", "title_all_caps"]
pr = ds1.groupby("label")[pats].mean().T[["real", "fraud"]] * 100
fig, ax = plt.subplots(figsize=(6.5, 3.2))
y = np.arange(len(pats))
ax.barh(y + 0.2, pr.real, height=0.38, color=REAL, label="real")
ax.barh(y - 0.2, pr.fraud, height=0.38, color=FRAUD, label="fraud")
ax.set_yticks(y, ["URL", "email", "phone", "money words", "all-caps title"]); ax.invert_yaxis()
ax.xaxis.set_major_formatter(mtick.PercentFormatter()); ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right"); ax.set_title("% of postings with each pattern (ds1)")
save("v2_patterns_by_class.png")
pr["fraud / real"] = (pr.fraud / pr.real).round(1)
pr.round(1)
''')

md("## 6. Employment type by class")
code(r'''
# fraud rate within each employment type (ds1 only)
et = ds1.groupby("employment_type").agg(postings=("fraudulent", "size"), fraud_rate=("fraudulent", "mean"))
et = et[et.postings >= 50].sort_values("fraud_rate")
base = ds1.fraudulent.mean()
fig, ax = plt.subplots(figsize=(6.5, 2.8))
ax.barh(et.index, et.fraud_rate * 100, color=[FRAUD if r > base else REAL for r in et.fraud_rate], height=0.55)
ax.axvline(base * 100, color=GRAY, ls="--", lw=1)
ax.text(base * 100 + 0.2, len(et) - 0.6, f"overall {base:.1%}", fontsize=8, color="#52514e")
ax.xaxis.set_major_formatter(mtick.PercentFormatter()); ax.grid(axis="y", visible=False)
ax.set_title("Fraud rate by employment type (ds1, types with 50+ postings)")
save("v2_fraud_rate_employment.png")
et.assign(fraud_rate=(et.fraud_rate * 100).round(1))
''')

md("## 7. Words that separate real and fraud")
code(r'''
# log-odds of each word appearing in fraud vs real descriptions (ds1, train split only, words in 20+ postings)
tr = ds1[ds1.split == "train"]
cv = CountVectorizer(stop_words="english", min_df=20, binary=True, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
X = cv.fit_transform(tr.title + " " + tr.description)
yv = tr.fraudulent.values
f_rate = (X[yv == 1].sum(0).A1 + 1) / ((yv == 1).sum() + 2)
r_rate = (X[yv == 0].sum(0).A1 + 1) / ((yv == 0).sum() + 2)
lo = pd.Series(np.log(f_rate / r_rate), index=cv.get_feature_names_out())
top = pd.DataFrame({"more common in fraud": lo.nlargest(15).index, "x": np.exp(lo.nlargest(15).values).round(1),
                    "more common in real": lo.nsmallest(15).index, "x ": np.exp(-lo.nsmallest(15).values).round(1)})
top
''')

md("## 8. Which features relate most to the target")
code(r'''
# mutual information between each engineered feature and the label (ds1, train split)
feats = ["has_company", "has_location", "has_salary", "has_experience", "has_employment_type",
         "has_url", "has_email", "has_phone", "money_words", "title_all_caps", "desc_words", "title_words"]
mi = pd.Series(mutual_info_classif(tr[feats], tr.fraudulent, discrete_features=[True]*10 + [False]*2,
                                   random_state=42), index=feats).sort_values()
corr = tr[feats + ["fraudulent"]].corr()["fraudulent"].drop("fraudulent")
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.barh(mi.index, mi.values, color=REAL, height=0.6)
ax.grid(axis="y", visible=False); ax.set_xlabel("Mutual information with fraudulent")
ax.set_title("Feature relevance (ds1 train split)")
save("v2_feature_relevance.png")
pd.DataFrame({"mutual_info": mi.round(4), "correlation": corr[mi.index].round(3)}).iloc[::-1]
''')

md("## 9. Source check: can a model tell the sources apart?")
code(r'''
# real postings only: if sources look very different, a model could learn "which source" instead of "fraud"
real = df[df.fraudulent == 0]
fig, ax = plt.subplots(figsize=(6.5, 3.0))
order = ["ds1_labeled", "linkedin", "techmap"]
ax.boxplot([real.loc[real.source == s, "desc_words"] for s in order], tick_labels=order, showfliers=False,
           patch_artist=True, widths=0.5, medianprops=dict(color="#0b0b0b"),
           boxprops=dict(facecolor="#b7d3f6", edgecolor=REAL))
ax.set_ylabel("Words in description"); ax.set_title("Description length of real postings by source")
save("v2_length_by_source.png")
real.groupby("source")[["desc_words", "has_company", "has_salary", "has_email", "money_words"]].mean().round(2)
''')

md("## 10. Why the synthetic Fake_Postings data was removed")
code(r'''
# compare the synthetic file to real fraud postings
syn = pd.read_csv(f"{RAW}/Fake_Postings.csv")
syn_words = syn.description.str.split().str.len()
template = syn.description.str.contains(r"Earn \$5000/week! Immediate hiring", regex=True).mean()
print(f"synthetic rows: {len(syn):,}; share with the same 'Earn $5000/week! Immediate hiring' line: {template:.0%}")
print(f"synthetic description length: median {syn_words.median():.0f} words (min {syn_words.min()}, max {syn_words.max()})")
print(f"real fraud description length (ds1): median {ds1[ds1.fraudulent==1].desc_words.median():.0f} words")
fig, ax = plt.subplots(figsize=(6.5, 2.4))
groups = [syn_words, ds1.loc[ds1.fraudulent == 1, "desc_words"], ds1.loc[ds1.fraudulent == 0, "desc_words"]]
bp = ax.boxplot(groups, orientation="horizontal", tick_labels=["synthetic Fake_Postings", "real fraud (ds1)", "real postings (ds1)"],
                showfliers=False, patch_artist=True, widths=0.5, medianprops=dict(color="#0b0b0b"))
for box, color in zip(bp["boxes"], (SYN, FRAUD, REAL)):
    box.set(facecolor=color, alpha=0.6, edgecolor=color)
ax.set_xscale("log"); ax.set_xlabel("Words in description (log scale)"); ax.grid(axis="y", visible=False)
ax.set_title("Synthetic 'fraud' looks nothing like real fraud")
save("v2_synthetic_vs_real_fraud.png")
syn[["title", "description", "company_profile", "location"]].head(3)
''')
code(r'''
# effect on the baseline (from src/models/train_baseline.py, ds1-only test set, cutoff 0.5)
res = pd.read_csv("../reports/baseline_results.csv")
res[(res.eval_set == "test_ds1_only")][["model", "precision", "recall", "f1", "pr_auc"]].round(3)
''')

md("""
## 11. Key findings
1. Fraud is rare: 3.4% of postings overall (4.3% within ds1). Accuracy is misleading, so we use precision, recall, F1 and PR-AUC.
2. Missing company info is the strongest single signal: 83% of real ds1 postings have a company profile vs 33% of fraud.
3. Fraud postings are shorter (median 201 vs 279 words) and more often contain phone numbers (3.3x), all-caps titles (3.6x), emails (2.1x) and URLs. Fraud lists a salary more often (28% vs 17%), and part-time jobs have the highest fraud rate (9.4%).
4. Many of the strongest fraud words are specific names (likely recruiters or companies that posted many scams), so a model can memorize them instead of learning general scam patterns. We split by company and will check performance on unseen companies.
5. Real postings from LinkedIn and Techmap are longer than ds1's, so we always report results on the ds1-only test set to avoid rewarding source detection.
6. The synthetic Fake_Postings file is one repeated template (about 14 words). Training with it cut recall on real scams from 87% to 50%, so it was removed.
""")

md("## Dataset card")
code(r'''
# print the dataset card so it is attached to the exported PDF
from IPython.display import Markdown, display
display(Markdown(open("../docs/dataset_card.md").read()))
''')

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/02_eda_v2.ipynb")
