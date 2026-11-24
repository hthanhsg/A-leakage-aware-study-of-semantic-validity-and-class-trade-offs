"""Evaluation on four axes: Utility - Fidelity - Validity - Privacy (+ cost, recorded by the runner)."""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
                             recall_score, roc_auc_score, roc_curve)

from .filtering import dcr, realism_scores


# ------------------------------------------------------------- Utility
def utility(y_true, P, classes, benign="Benign", rare=None, holdout=None):
    """y_true may contain classes absent from `classes` (holdout/cross-dataset) -> handled separately."""
    classes = list(classes)
    cid = {c: i for i, c in enumerate(classes)}
    y_true = np.asarray(y_true, dtype=object)
    pred = P.argmax(1)
    known = np.array([v in cid for v in y_true])
    out = {}
    if known.any():
        yt = np.array([cid[v] for v in y_true[known]])
        pk, Pk = pred[known], P[known]
        labels = sorted(set(yt))
        out.update(
            accuracy=accuracy_score(yt, pk),
            macro_f1=f1_score(yt, pk, average="macro", labels=labels, zero_division=0),
            weighted_f1=f1_score(yt, pk, average="weighted", labels=labels, zero_division=0),
            mcc=matthews_corrcoef(yt, pk),
        )
        rec = recall_score(yt, pk, average=None, labels=labels, zero_division=0)
        out["recall_per_class"] = {classes[l]: float(r) for l, r in zip(labels, rec)}
        f1c = f1_score(yt, pk, average=None, labels=labels, zero_division=0)
        out["f1_per_class"] = {classes[l]: float(r) for l, r in zip(labels, f1c)}
        aps = []
        for l in labels:
            if 0 < (yt == l).sum() < len(yt):
                aps.append(average_precision_score(yt == l, Pk[:, l]))
        out["macro_pr_auc"] = float(np.mean(aps)) if aps else np.nan
        f1d = out["f1_per_class"]
        att = [v for c, v in f1d.items() if c != benign]
        out["macro_f1_attacks"] = float(np.mean(att)) if att else np.nan  # attack classes only
        com = [v for c, v in f1d.items() if c != benign and c not in (rare or [])]
        out["macro_f1_common"] = float(np.mean(com)) if com else np.nan  # non-rare attack classes
        if rare:
            rr = [out["recall_per_class"][c] for c in rare if c in out["recall_per_class"]]
            out["rare_recall"] = float(np.mean(rr)) if rr else np.nan
            rf = [out["f1_per_class"][c] for c in rare if c in out["f1_per_class"]]
            out["rare_f1"] = float(np.mean(rf)) if rf else np.nan
    # Binary detection: attack = every class other than Benign
    if benign in cid:
        is_att = y_true != benign
        score = 1.0 - P[:, cid[benign]]
        pred_att = pred != cid[benign]
        if is_att.any() and (~is_att).any():
            out["binary_tpr"] = float(pred_att[is_att].mean())
            out["binary_fpr"] = float(pred_att[~is_att].mean())
            out["binary_roc_auc"] = float(roc_auc_score(is_att, score))
            fpr, tpr, _ = roc_curve(is_att, score)
            out["tpr_at_fpr1"] = float(np.interp(0.01, fpr, tpr))
    if holdout is not None:
        m = y_true == holdout
        if m.any() and benign in cid:
            out["holdout_detection"] = float((pred[m] != cid[benign]).mean())
    return out


# ------------------------------------------------------------ Fidelity
def fidelity(Z_real, y_real, Z_syn, y_syn, n_num, seed=0, max_n=20000):
    """Compare syn with real under the SAME class distribution (real is sampled with syn class ratios)."""
    if len(Z_syn) == 0:
        return {}
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y_syn):
        pool = np.where(y_real == c)[0]
        k = min(int((y_syn == c).sum()), len(pool))
        if k:
            idx.append(rng.choice(pool, k, replace=False))
    R = Z_real[np.concatenate(idx)] if idx else Z_real[:0]
    S = Z_syn[rng.choice(len(Z_syn), min(len(Z_syn), max_n), replace=False)]
    R = R[rng.choice(len(R), min(len(R), max_n), replace=False)]
    wd = [wasserstein_distance(R[:, j], S[:, j]) for j in range(n_num)] if n_num else [np.nan]
    with np.errstate(invalid="ignore", divide="ignore"):  # constant column within a class -> NaN -> 0
        cr = np.nan_to_num(np.corrcoef(R, rowvar=False)); cs = np.nan_to_num(np.corrcoef(S, rowvar=False))
    iu = np.triu_indices_from(cr, 1)
    _, auc = realism_scores(R, S, seed=seed)
    return {"wasserstein_mean": float(np.mean(wd)), "corr_diff": float(np.abs(cr - cs)[iu].mean()),
            "c2st_auc": float(auc)}


# ------------------------------------------------------------- Privacy
def privacy(Z_tr, Z_va, Z_syn, seed=0, max_n=5000):
    if len(Z_syn) == 0:
        return {}
    rng = np.random.default_rng(seed)
    S = Z_syn[rng.choice(len(Z_syn), min(len(Z_syn), 4 * max_n), replace=False)]
    n = min(len(Z_tr), len(Z_va), max_n)
    mem = Z_tr[rng.choice(len(Z_tr), n, replace=False)]
    non = Z_va[rng.choice(len(Z_va), n, replace=False)]
    d_syn = dcr(S[:max_n], Z_tr)
    d_ref = dcr(non, Z_tr)
    s_mem, s_non = -dcr(mem, S), -dcr(non, S)  # distance-based MIA
    auc = roc_auc_score(np.r_[np.ones(n), np.zeros(n)], np.r_[s_mem, s_non])
    # Many real flows coincide in encoded space, so the median real->train distance can be 0; the ratio is
    # therefore computed on strictly positive distances, and exact copies are reported separately.
    pos_ref = d_ref[d_ref > 1e-9]
    ref_med = float(np.median(pos_ref)) if len(pos_ref) else np.nan
    return {"dcr_median": float(np.median(d_syn)),
            "dcr_ratio": float(np.median(d_syn) / ref_med) if ref_med and np.isfinite(ref_med) else np.nan,
            "dcr_frac_exact_copy": float(np.mean(d_syn <= 1e-9)),
            "real_frac_exact_copy": float(np.mean(d_ref <= 1e-9)),
            "dcr_frac_below_ref5": float(np.mean(d_syn < np.percentile(d_ref, 5))), "mia_auc": float(auc)}
