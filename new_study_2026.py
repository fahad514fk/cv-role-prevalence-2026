"""2026 follow-up study to Shah Fahad's MSc dissertation (resume classification). NEW data, NEW study.

This is NOT the dissertation's data. It uses the public Djinni Recruitment Dataset (English candidate CVs,
MIT licence; Drushchak & Romanyshyn, UNLP 2024) to rebuild a clean five-role benchmark, then:
  1. trains Naive Bayes, SVM (TF-IDF) and fine-tuned DistilBERT with the dissertation's settings and seed 42 on a
     de-duplicated split (no exact or near duplicates anywhere in the sample, so none across splits);
  2. runs a class-prevalence experiment after Keith & O'Connor (EMNLP 2018): held-out batches with varied role mixes,
     comparing classify-and-count (CC), adjusted CC (ACC), probabilistic CC (PCC) and, as extras, probabilistic
     ACC (PACC) and the Saerens et al. (2002) EM prior adjustment.

Usage (from this folder):
    python3 new_study_2026.py            # everything: download (237 MB, SHA-256 checked), build, all 3 models,
                                         # prevalence experiment. DistilBERT takes ~35-45 min on CPU.
    python3 new_study_2026.py --quick    # skip DistilBERT training (reuses saved DistilBERT outputs if present)

Outputs: data/five_roles_2026.parquet, results/dataset_build.json, results/new_study_results.json,
results/distilbert_outputs_seed42.npz.
"""
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data_raw", "djinni", "candidates_en.parquet")
RAW_URL = ("https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english/resolve/"
           "86255174c6c378a2b52cd7e81d79002c64d899a6/data/train-00000-of-00001.parquet")
RAW_SHA256 = "a7b1378e0b231507c377659bf2c80fa4ba60c4e31e1107b13425d103879ecc3b"
DATA_OUT = os.path.join(HERE, "data", "five_roles_2026.parquet")
RES = os.path.join(HERE, "results")
BERT_CACHE = os.path.join(RES, "distilbert_outputs_seed42.npz")

SEED = 42
QUICK = "--quick" in sys.argv
ROLES = ["Data Analyst", "Java Developer", "Python Developer", "Software Tester", "Website Developer/Designer"]
POOL_CAP = 2000          # per role, screened for duplicates before sampling
PER_ROLE = 600           # distinct CVs per role in the final sample
SPLIT = {"train": 200, "val": 100, "test": 300}   # per role
MIN_TOKENS = 50          # minimum cleaned tokens for a CV to be kept
NEAR_DUP_COSINE = 0.90   # TF-IDF cosine at or above which two CVs count as near duplicates
ROLE_WORDS = ["python", "java", "javascript", "analyst", "analytics", "qa", "tester", "testing", "test", "web",
              "website", "frontend", "front", "designer", "design"]   # cleaned-text forms hidden in the masking check
WEB_TITLE = re.compile(r"\b(web|front[- ]?end|frontend|html|wordpress|webflow)\b", re.I)


# ---------------------------------------------------------------- text cleaning (identical to the dissertation's
# utils/preprocessing.py clean_text, copied here so this study is self-contained)
def _nltk():
    import nltk
    for pkg, path in (("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    return set(stopwords.words("english")), WordNetLemmatizer()


STOP, LEMMA = _nltk()


def clean_text(text):
    if not isinstance(text, str):
        return ""
    words = re.sub(r"[^a-zA-Z]", " ", text.lower()).split()
    return " ".join(LEMMA.lemmatize(w) for w in words if w not in STOP and len(w) > 2)


# ---------------------------------------------------------------- 1. dataset
def download():
    if not os.path.exists(RAW):
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        print("Downloading Djinni English candidate CVs (237 MB) ...")
        urllib.request.urlretrieve(RAW_URL, RAW)
    digest = hashlib.sha256()
    with open(RAW, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != RAW_SHA256:
        sys.exit(f"SHA-256 mismatch for {RAW}; delete it and rerun.")


def role_of(keyword, position):
    if keyword == "Python":
        return "Python Developer"
    if keyword == "Java":
        return "Java Developer"
    if keyword == "Data Analyst":
        return "Data Analyst"
    if keyword in ("QA", "QA Automation"):
        return "Software Tester"
    if keyword in ("JavaScript", "PHP", "Design") and WEB_TITLE.search(position or ""):
        return "Website Developer/Designer"
    return None


def near_duplicate_drop(texts, threshold):
    """Greedy near-duplicate removal: walk in order, drop a CV if an earlier KEPT CV has cosine >= threshold."""
    X = TfidfVectorizer(sublinear_tf=True, min_df=2).fit_transform(texts)
    n, keep = X.shape[0], np.ones(X.shape[0], dtype=bool)
    sens = {t: 0 for t in (0.80, 0.90, 0.95)}       # CVs having any earlier neighbour at or above t
    for start in range(0, n, 1000):
        sims = (X[start:start + 1000] @ X.T).toarray()
        for r in range(sims.shape[0]):
            i = start + r
            earlier = sims[r, :i]
            for t in sens:
                sens[t] += bool((earlier >= t).any())
            if (earlier[keep[:i]] >= threshold).any():
                keep[i] = False
    return keep, sens


def build_dataset():
    download()
    raw = pd.read_parquet(RAW, columns=["id", "Position", "Primary Keyword", "CV"])
    raw["role"] = [role_of(k, p) for k, p in zip(raw["Primary Keyword"], raw["Position"])]
    pool = raw[raw["role"].notna()].copy()
    info = {"source_rows": len(raw), "five_role_rows": len(pool),
            "five_role_rows_by_role": pool["role"].value_counts().sort_index().to_dict()}

    pool["cleaned_text"] = pool["CV"].map(clean_text)
    pool["n_tokens"] = pool["cleaned_text"].str.split().str.len()
    short = pool["n_tokens"] < MIN_TOKENS
    info["removed_too_short"] = int(short.sum())
    pool = pool[~short]

    # Exact duplicates over the whole five-role pool (on the cleaned text, so case/punctuation changes still count)
    exact = pool.duplicated("cleaned_text", keep="first")
    info["removed_exact_duplicates_full_pool"] = int(exact.sum())
    pool = pool[~exact]

    # Cap each role at POOL_CAP CVs (random, seeded), then screen that pool for near duplicates across all roles
    pool = (pool.sample(frac=1, random_state=SEED).groupby("role", group_keys=False).head(POOL_CAP)
            .sample(frac=1, random_state=SEED).reset_index(drop=True))
    keep, sens = near_duplicate_drop(pool["cleaned_text"].tolist(), NEAR_DUP_COSINE)
    info["screened_pool"] = len(pool)
    info["screened_pool_by_role"] = pool["role"].value_counts().sort_index().to_dict()
    info["removed_near_duplicates"] = int((~keep).sum())
    info["removed_near_duplicates_by_role"] = pool.loc[~keep, "role"].value_counts().sort_index().to_dict()
    info["near_dup_threshold_sensitivity"] = {f"cos>={t}": v for t, v in sens.items()}
    pool = pool[keep]

    sample = pool.groupby("role").sample(PER_ROLE, random_state=SEED).reset_index(drop=True)
    web = sample[sample["role"] == "Website Developer/Designer"]["Primary Keyword"].value_counts().to_dict()
    tester = sample[sample["role"] == "Software Tester"]["Primary Keyword"].value_counts().to_dict()
    info.update({"final_sample": len(sample), "final_per_role": PER_ROLE,
                 "website_role_composition": web, "software_tester_composition": tester,
                 "median_tokens_by_role": sample.groupby("role")["n_tokens"].median().to_dict(),
                 "synthetic_resumes": 0})
    os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
    sample[["id", "role", "Primary Keyword", "Position", "cleaned_text", "n_tokens"]].to_parquet(DATA_OUT,
                                                                                               index=False)
    return sample, info


def split(sample, seed):
    sample = sample.copy()
    sample["label"] = sample["role"].map({r: i for i, r in enumerate(ROLES)})
    parts = {"train": [], "val": [], "test": []}
    for _, g in sample.groupby("role"):
        g = g.sample(frac=1, random_state=seed)
        a, b = SPLIT["train"], SPLIT["train"] + SPLIT["val"]
        parts["train"].append(g.iloc[:a]); parts["val"].append(g.iloc[a:b]); parts["test"].append(g.iloc[b:])
    return {k: pd.concat(v).reset_index(drop=True) for k, v in parts.items()}


# ---------------------------------------------------------------- 2. classifiers
def doc_scores(y, pred, boot=1000):
    rng = np.random.default_rng(SEED)
    acc, f1s = [], []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y))
        acc.append(accuracy_score(y[i], pred[i])); f1s.append(f1_score(y[i], pred[i], average="macro"))
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "accuracy_95ci": [round(float(np.percentile(acc, q)), 4) for q in (2.5, 97.5)],
            "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "macro_f1_95ci": [round(float(np.percentile(f1s, q)), 4) for q in (2.5, 97.5)],
            "per_role_f1": {r: round(v, 4) for r, v in zip(ROLES, f1_score(y, pred, average=None))},
            "confusion_matrix_rows_true_cols_pred": confusion_matrix(y, pred).tolist(), "n": int(len(y))}


def tfidf_models(parts):
    vec = TfidfVectorizer(max_features=5000)
    Xtr = vec.fit_transform(parts["train"]["cleaned_text"])
    out = {}
    for name, model in (("Naive Bayes", MultinomialNB()),
                        ("SVM", SVC(class_weight="balanced", probability=True, random_state=SEED))):
        model.fit(Xtr, parts["train"]["label"])
        out[name] = {}
        for k in ("val", "test"):
            X = vec.transform(parts[k]["cleaned_text"])
            out[name][k] = (model.predict(X), model.predict_proba(X))
    return out


def distilbert(parts):
    import torch
    from datasets import Dataset
    from transformers import (DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer,
                              TrainingArguments, set_seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def enc(frame):
        ds = Dataset.from_pandas(frame[["cleaned_text", "label"]].reset_index(drop=True))
        ds = ds.map(lambda b: tok(b["cleaned_text"], truncation=True, padding="max_length", max_length=256),
                    batched=True)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        return ds

    set_seed(SEED)
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(ROLES))
    args = TrainingArguments(output_dir=os.path.join(RES, "bert_tmp"), num_train_epochs=3,
                             per_device_train_batch_size=8, per_device_eval_batch_size=32, save_strategy="no",
                             logging_steps=25, report_to=[], seed=SEED, use_cpu=True)
    trainer = Trainer(model=model, args=args, train_dataset=enc(parts["train"]))
    started = time.time()
    trainer.train()
    minutes = round((time.time() - started) / 60, 1)
    logits = {k: trainer.predict(enc(parts[k])).predictions for k in ("val", "test")}
    np.savez(BERT_CACHE, val=logits["val"], test=logits["test"], minutes=minutes,
             val_ids=parts["val"]["id"].to_numpy(), test_ids=parts["test"]["id"].to_numpy())


def load_bert(parts):
    if not os.path.exists(BERT_CACHE):
        return None, None
    z = np.load(BERT_CACHE, allow_pickle=True)
    if list(z["test_ids"]) != list(parts["test"]["id"]) or list(z["val_ids"]) != list(parts["val"]["id"]):
        print("Saved DistilBERT outputs do not match this split; ignoring them.")
        return None, None
    out = {}
    for k in ("val", "test"):
        e = np.exp(z[k] - z[k].max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        out[k] = (p.argmax(1), p)
    return out, float(z["minutes"])


# ---------------------------------------------------------------- 3. prevalence estimation
def simplex_lsq(M, q):
    """argmin ||M^T pi - q||^2 subject to pi >= 0, sum(pi) = 1."""
    K = len(q)
    res = minimize(lambda p: np.sum((M.T @ p - q) ** 2), np.full(K, 1 / K), method="SLSQP",
                   bounds=[(0, 1)] * K, constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1}])
    p = np.clip(res.x, 0, None)
    return p / p.sum()


def em_saerens(P, train_prior, iters=200, tol=1e-6):
    pi = train_prior.copy()
    for _ in range(iters):
        w = P * (pi / train_prior)
        w /= w.sum(1, keepdims=True)
        new = w.mean(0)
        if np.abs(new - pi).max() < tol:
            return new
        pi = new
    return pi


def prevalence_experiment(preds, parts, batch_sizes=(50, 200), n_batches=500):
    K = len(ROLES)
    yv, yt = parts["val"]["label"].to_numpy(), parts["test"]["label"].to_numpy()
    by_class = [np.flatnonzero(yt == k) for k in range(K)]
    train_prior = np.bincount(parts["train"]["label"], minlength=K) / len(parts["train"])
    rng = np.random.default_rng(SEED)
    batches = []                                         # identical batches for every model and method
    for size in batch_sizes:
        for _ in range(n_batches):
            pi = rng.dirichlet(np.ones(K))
            counts = rng.multinomial(size, pi)
            idx = np.concatenate([rng.choice(by_class[k], c, replace=False) for k, c in enumerate(counts)])
            batches.append((size, idx, counts / size))

    out = {}
    for model, p in preds.items():
        (hv, Pv), (ht, Pt) = p["val"], p["test"]
        M_hard = np.vstack([np.bincount(hv[yv == k], minlength=K) / (yv == k).sum() for k in range(K)])
        M_soft = np.vstack([Pv[yv == k].mean(0) for k in range(K)])
        rows = []
        for size, idx, true in batches:
            cc = np.bincount(ht[idx], minlength=K) / len(idx)
            pcc = Pt[idx].mean(0)
            est = {"CC": cc, "ACC": simplex_lsq(M_hard, cc), "PCC": pcc, "PACC": simplex_lsq(M_soft, pcc),
                   "EM (Saerens 2002)": em_saerens(Pt[idx], train_prior)}
            shift = 0.5 * np.abs(true - train_prior).sum()
            rows.append({"size": size, "shift": shift, **{m: np.abs(e - true).mean() for m, e in est.items()}})
        df = pd.DataFrame(rows)
        df["shift_bin"] = pd.cut(df["shift"], [0, 0.2, 0.4, 1.0], labels=["low (<0.2)", "mid (0.2-0.4)",
                                                                          "high (>=0.4)"], include_lowest=True)
        methods = ["CC", "ACC", "PCC", "PACC", "EM (Saerens 2002)"]
        summary = {"overall": {m: round(df[m].mean(), 4) for m in methods},
                   "overall_sd": {m: round(df[m].std(), 4) for m in methods},
                   "by_batch_size": {int(s): {m: round(g[m].mean(), 4) for m in methods}
                                     for s, g in df.groupby("size")},
                   "by_shift": {str(b): {"n_batches": int(len(g)), **{m: round(g[m].mean(), 4) for m in methods}}
                                for b, g in df.groupby("shift_bin", observed=True)},
                   "share_of_batches_where_ACC_beats_CC": round(float((df["ACC"] < df["CC"]).mean()), 4)}
        out[model] = summary
    return {"design": {"test_pool_per_role": SPLIT["test"], "batch_sizes": list(batch_sizes),
                       "batches_per_size": n_batches, "true_mix": "Dirichlet(1,1,1,1,1), i.e. uniform over mixes",
                       "training_mix": dict(zip(ROLES, train_prior.round(3).tolist())),
                       "confusion_rates_from": "validation split (100 per role), never the test pool",
                       "error": "mean absolute error of the 5 estimated shares, averaged over batches",
                       "shift": "total variation distance between batch mix and training mix"},
            "results_by_model": out}


# ---------------------------------------------------------------- main
def main():
    os.makedirs(RES, exist_ok=True)
    sample, info = build_dataset()
    with open(os.path.join(RES, "dataset_build.json"), "w", encoding="utf-8") as h:
        json.dump(info, h, indent=2, default=str)
    print(json.dumps(info, indent=2, default=str))

    parts = split(sample, SEED)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(parts[a]["cleaned_text"]) & set(parts[b]["cleaned_text"]), "duplicate across splits"
    results = {"study": "2026 follow-up to the MSc dissertation; new data (Djinni), not the dissertation's data",
               "data": info, "split_per_role": SPLIT, "seed": SEED, "models": {}}

    preds = tfidf_models(parts)
    if not QUICK:
        distilbert(parts)
    bert, minutes = load_bert(parts)
    if bert is not None:
        preds["DistilBERT"] = bert
    for name, p in preds.items():
        results["models"][name] = {"test": doc_scores(parts["test"]["label"].to_numpy(), p["test"][0])}
        if name == "DistilBERT":
            results["models"][name]["training_minutes_cpu"] = minutes
        print(f"{name:11} acc {results['models'][name]['test']['accuracy']}  "
              f"macro-F1 {results['models'][name]['test']['macro_f1']}")

    # Robustness of the TF-IDF models to the split (DistilBERT is run once, seed 42 only, because of CPU cost)
    seeds = {}
    for s in range(SEED, SEED + 5):
        ps = split(sample, s)
        for name, p in tfidf_models(ps).items():
            y = ps["test"]["label"].to_numpy()
            seeds.setdefault(name, []).append((accuracy_score(y, p["test"][0]),
                                               f1_score(y, p["test"][0], average="macro")))
    results["tfidf_split_seeds_42_to_46"] = {
        n: {"accuracy_mean": round(float(np.mean([a for a, _ in v])), 4),
            "accuracy_sd": round(float(np.std([a for a, _ in v], ddof=1)), 4),
            "macro_f1_mean": round(float(np.mean([f for _, f in v])), 4),
            "macro_f1_sd": round(float(np.std([f for _, f in v], ddof=1)), 4)} for n, v in seeds.items()}

    # Robustness: hide the obvious role words (after cleaning) and retrain the TF-IDF models
    mask = re.compile(r"\b(" + "|".join(ROLE_WORDS) + r")\b")
    masked = {k: v.assign(cleaned_text=v["cleaned_text"].str.replace(mask, " ", regex=True)) for k, v in parts.items()}
    results["tfidf_role_words_masked"] = {
        "masked_words": ROLE_WORDS,
        **{n: {"accuracy": round(accuracy_score(masked["test"]["label"], p["test"][0]), 4),
               "macro_f1": round(f1_score(masked["test"]["label"], p["test"][0], average="macro"), 4)}
           for n, p in tfidf_models(masked).items()}}

    results["prevalence"] = prevalence_experiment(preds, parts)
    for m, s in results["prevalence"]["results_by_model"].items():
        print(f"Prevalence MAE {m:11} {s['overall']}")

    # Prevalence under a skewed TRAINING mix (as in Keith & O'Connor's shifted setting): one role makes up 2/3 of
    # training (200 CVs), the other four 25 each; repeated with each role as the dominant one. TF-IDF models only.
    skew = {}
    for dom in range(len(ROLES)):
        tr = parts["train"]
        sub = pd.concat([tr[tr["label"] == dom]] +
                        [tr[tr["label"] == k].sample(25, random_state=SEED) for k in range(len(ROLES)) if k != dom])
        sp = {**parts, "train": sub.reset_index(drop=True)}
        for n, s in prevalence_experiment(tfidf_models(sp), sp)["results_by_model"].items():
            skew.setdefault(n, []).append(s["overall"])
    results["prevalence_skewed_training"] = {
        "design": "train = 200 CVs of one role + 25 of each other role (shares 0.67 / 0.083); same test batches; "
                  "averaged over the 5 choices of dominant role",
        "results_by_model": {n: {m: round(float(np.mean([d[m] for d in v])), 4) for m in v[0]}
                             for n, v in skew.items()}}
    for m, s in results["prevalence_skewed_training"]["results_by_model"].items():
        print(f"Skewed-train prevalence MAE {m:11} {s}")
    results["run"] = {"quick": QUICK, "distilbert_included": bert is not None,
                      "finished": time.strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(RES, "new_study_results.json"), "w", encoding="utf-8") as h:
        json.dump(results, h, indent=2, default=str)
    print("Saved results/new_study_results.json")


if __name__ == "__main__":
    main()
