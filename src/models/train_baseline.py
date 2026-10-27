"""Train and evaluate baseline fraud classifiers on the v2 dataset.

Models:
    * majority  - always predicts "real" (shows why accuracy misleads)
    * rule      - flags postings with no company info or an email in the text
    * tfidf_lr  - TF-IDF (title + description) + engineered features,
                  logistic regression with balanced class weights
    * tfidf_lr_synthetic - same model trained with the synthetic Fake Postings
                  rows added (comparison only; same val/test sets)

Every run is logged to Weights & Biases (offline by default; upload later with
``wandb sync wandb/offline-run-*``) and to ``reports/baseline_results.csv``.

Example:
    python src/models/train_baseline.py --data data/processed/v2
"""

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             log_loss, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
NUM_COLS = ["has_company", "has_location", "has_salary", "has_experience",
            "has_employment_type", "has_url", "has_email", "has_phone",
            "money_words", "title_all_caps", "desc_words", "title_words"]
CAT_COLS = ["employment_type"]


def load(path):
    """Read a split and build the combined text column."""
    df = pd.read_csv(path)
    df["text"] = df["title"].fillna("") + " . " + df["description"].fillna("")
    return df


def build_pipeline(C=1.0, ngram=(1, 2), min_df=2, max_features=50000):
    """TF-IDF on text + scaled numeric features + one-hot employment type -> LR.

    Args:
        C: Inverse regularization strength (smaller = stronger regularization).
        ngram: Word n-gram range for TF-IDF.
        min_df: Ignore words that appear in fewer than this many postings.
        max_features: Vocabulary size cap.

    Returns:
        Unfitted sklearn Pipeline. The model minimizes log loss (cross-entropy).
    """
    log_len = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    features = ColumnTransformer([
        ("tfidf", TfidfVectorizer(ngram_range=ngram, min_df=min_df, max_features=max_features,
                                  sublinear_tf=True, lowercase=True, stop_words="english"), "text"),
        ("num", Pipeline([("log", log_len), ("scale", StandardScaler())]), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, solver="liblinear",
                             random_state=SEED)
    return Pipeline([("features", features), ("clf", clf)])


def rule_scores(df):
    """Simple rule: fraud if no company info or an email address in the text."""
    return ((df["has_company"] == 0) | (df["has_email"] == 1)).astype(float).values


def metrics(y, p, threshold=0.5):
    """Precision, recall, F1, PR-AUC, ROC-AUC, log loss and confusion matrix."""
    pred = (p >= threshold).astype(int)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "pr_auc": average_precision_score(y, p),
            "roc_auc": roc_auc_score(y, p),
            "log_loss": log_loss(y, pc, labels=[0, 1]),
            "accuracy": (tp + tn) / len(y),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def best_threshold(y, p):
    """Threshold that maximizes F1 on the validation set."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def evaluate_sets(name, score_fn, sets, threshold):
    """Score val / test / test_ds1_only and return rows of metrics."""
    rows = []
    for set_name, df in sets.items():
        m = metrics(df["fraudulent"].values, score_fn(df), threshold)
        rows.append({"model": name, "eval_set": set_name, "threshold": threshold, **m})
    return rows


def pr_curve_plot(curves, path, title):
    """Save precision-recall curves for several models on one chart."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, lw=2, label=f"{label} (PR-AUC {average_precision_score(y, p):.2f})")
    ax.axhline(np.mean(y), color="#9a998f", ls="--", lw=1, label="random guess")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title, loc="left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/processed/v2", type=Path)
    ap.add_argument("--reports", default="reports", type=Path)
    ap.add_argument("--models", default="models", type=Path)
    ap.add_argument("--project", default="jobai-fraud")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("WANDB_MODE", "offline")
    args.reports.mkdir(parents=True, exist_ok=True); args.models.mkdir(exist_ok=True)
    (args.reports / "figures" / "v2").mkdir(parents=True, exist_ok=True)

    train, val, test = (load(args.data / f"{s}.csv") for s in ("train", "val", "test"))
    sets = {"val": val, "test": test, "test_ds1_only": test[test.source == "ds1_labeled"]}
    ytr = train["fraudulent"].values
    results, runs = [], []

    def log_run(name, config, rows, fit_seconds=None):
        """Log one run to W&B (if enabled) and keep it for the CSV."""
        runs.append({"run": name, **config, "fit_seconds": fit_seconds})
        if args.no_wandb:
            return
        import wandb
        run = wandb.init(project=args.project, name=name, config=config, reinit=True,
                         dir=str(args.reports))
        for r in rows:
            run.log({f"{r['eval_set']}/{k}": v for k, v in r.items()
                     if k not in ("model", "eval_set")})
        if fit_seconds is not None:
            run.log({"fit_seconds": fit_seconds})
        run.finish()

    # 1. majority class: predicts probability = training fraud rate for everyone
    prior = ytr.mean()
    rows = evaluate_sets("majority", lambda d: np.full(len(d), prior), sets, 0.5)
    results += rows; log_run("majority", {"model": "majority"}, rows)

    # 2. rule baseline
    rows = evaluate_sets("rule", rule_scores, sets, 0.5)
    results += rows; log_run("rule", {"model": "rule"}, rows)

    # 3. small sweep of TF-IDF + LR configs (also gives the timing estimate)
    grid = [dict(C=c, ngram=ng) for c in (0.3, 1.0, 3.0) for ng in ((1, 1), (1, 2))]
    sweep = []
    for cfg in grid:
        pipe = build_pipeline(**cfg)
        t0 = time.perf_counter(); pipe.fit(train, ytr); secs = time.perf_counter() - t0
        pv = pipe.predict_proba(val)[:, 1]
        m = metrics(val["fraudulent"].values, pv)
        name = f"tfidf_lr_C{cfg['C']}_ng{cfg['ngram'][1]}"
        sweep.append({"run": name, **cfg, "val_pr_auc": m["pr_auc"], "val_f1": m["f1"],
                      "fit_seconds": secs})
        log_run(name, {"model": "tfidf_lr", "C": cfg["C"], "ngram_max": cfg["ngram"][1],
                       "sweep": True},
                [{"eval_set": "val", **m}], secs)
        print(f"{name}: val PR-AUC {m['pr_auc']:.3f}  F1 {m['f1']:.3f}  fit {secs:.1f}s")
    sweep_df = pd.DataFrame(sweep)
    best = sweep_df.sort_values("val_pr_auc", ascending=False).iloc[0]
    best_cfg = dict(C=float(best["C"]), ngram=tuple(best["ngram"]))

    # 4. final baseline with the best config; threshold tuned on val
    pipe = build_pipeline(**best_cfg)
    t0 = time.perf_counter(); pipe.fit(train, ytr); secs = time.perf_counter() - t0
    thr = best_threshold(val["fraudulent"].values, pipe.predict_proba(val)[:, 1])
    score = lambda d: pipe.predict_proba(d)[:, 1]
    for t, tag in ((0.5, "tfidf_lr"), (thr, "tfidf_lr_tuned_threshold")):
        rows = evaluate_sets(tag, score, sets, t)
        results += rows
        log_run(tag, {"model": "tfidf_lr", **{k: str(v) for k, v in best_cfg.items()},
                      "threshold": t}, rows, secs)
    joblib.dump(pipe, args.models / "tfidf_lr_baseline.joblib")

    # 5. comparison: same model trained with the synthetic rows added
    syn_path = args.data / "train_with_synthetic.csv"
    if syn_path.exists():
        tr_syn = load(syn_path)
        pipe_syn = build_pipeline(**best_cfg)
        t0 = time.perf_counter(); pipe_syn.fit(tr_syn, tr_syn["fraudulent"].values)
        secs_syn = time.perf_counter() - t0
        score_syn = lambda d: pipe_syn.predict_proba(d)[:, 1]
        rows = evaluate_sets("tfidf_lr_with_synthetic", score_syn, sets, 0.5)
        results += rows
        log_run("tfidf_lr_with_synthetic", {"model": "tfidf_lr", "train": "with_synthetic"},
                rows, secs_syn)
        # how the synthetic-trained model does on synthetic-looking text vs real scams
        syn_rows = tr_syn[tr_syn.source == "synthetic_fake_postings"].sample(1000, random_state=SEED)
        print("synthetic-trained model flags", (score_syn(syn_rows) >= .5).mean(),
              "of held-in synthetic rows")

    # 6. save tables and charts
    res = pd.DataFrame(results)
    res.to_csv(args.reports / "baseline_results.csv", index=False)
    sweep_df.to_csv(args.reports / "baseline_sweep.csv", index=False)
    d1 = sets["test_ds1_only"]
    curves = {"TF-IDF + LR": (d1["fraudulent"].values, score(d1)),
              "Rule": (d1["fraudulent"].values, rule_scores(d1))}
    if syn_path.exists():
        curves["TF-IDF + LR (trained with synthetic)"] = (d1["fraudulent"].values, score_syn(d1))
    pr_curve_plot(curves, args.reports / "figures" / "v2" / "pr_curve_test_ds1.png",
                  "Precision-recall on real labeled test postings")

    # top words pushing toward fraud / real (for interpretation)
    names = pipe.named_steps["features"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(coefs)
    top = pd.DataFrame({"toward_fraud": names[order[::-1][:20]],
                        "toward_real": names[order[:20]]})
    top.to_csv(args.reports / "top_features.csv", index=False)

    summary = {"best_config": {k: str(v) for k, v in best_cfg.items()},
               "threshold_from_val": thr, "fit_seconds_final": secs,
               "sweep_fit_seconds_mean": float(sweep_df.fit_seconds.mean()),
               "train_rows": len(train), "train_fraud_rate": float(prior)}
    (args.reports / "baseline_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print(res[["model", "eval_set", "precision", "recall", "f1", "pr_auc", "roc_auc",
               "accuracy", "tp", "fp", "fn"]].round(3).to_string(index=False))
    print(json.dumps(summary, indent=2)); print(top.to_string(index=False))


if __name__ == "__main__":
    main()
