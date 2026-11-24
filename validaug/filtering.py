"""Three-criterion filter: realistic - non-memorized - useful (Sec. 2.2 / pipeline figure).

(a) Realistic      : real/synthetic discriminator (XGBoost, K-fold cross-fitting so synthetic scores
                     are out-of-fold). The REAL sample is drawn with the class distribution of the
                     synthetic set (class-matched), otherwise the discriminator simply learns
                     "minority class vs Benign" instead of "real vs synthetic". Within each class,
                     the fraction rho of samples with the lowest p_real is removed (rank-based:
                     XGBoost C2ST on tabular data is often >= 0.9, so an absolute threshold on p_real
                     is unstable across datasets).
(b) Non-memorized  : DCR (distance to the closest same-class training record) must be >= the q-th
                     percentile of the "natural" DCR (real val -> train). If the class is absent from
                     val, the 2nd-nearest-neighbour distance within train is used.
(c) Useful         : ensemble margin m_y in (0, m_hi], m_hi = 90th percentile of the real training
                     margins of the same class (removes "too easy" samples and wrong-side samples).
                     Safeguard: if this keeps < min_keep_frac of a class (weak ensemble on a very rare
                     class), the top min_keep_frac by margin is kept instead (reported as margin_relaxed).
Ranking score used by the budget: score = -|m_y - delta|.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from .utils import LOG


def class_matched_real(Z_real, y_real, y_syn, seed=0, max_real=50_000, min_real=1000):
    """Sample real rows with the class proportions of `y_syn` (classes missing in real are skipped)."""
    rng = np.random.default_rng(seed)
    y_real = np.asarray(y_real, dtype=object)
    y_syn = np.asarray(y_syn, dtype=object)
    n_total = int(min(max(len(y_syn), min_real), max_real, len(Z_real)))
    cls, cnt = np.unique(y_syn, return_counts=True)
    idx = []
    for c, k in zip(cls, cnt):
        pool = np.where(y_real == c)[0]
        if len(pool) == 0:
            continue
        want = int(round(n_total * k / len(y_syn)))
        idx.append(rng.choice(pool, min(want, len(pool)), replace=False) if want else pool[:0])
    if not idx:
        return Z_real[rng.choice(len(Z_real), n_total, replace=False)]
    return Z_real[np.concatenate(idx)]


def realism_scores(Z_real, Z_syn, k=3, seed=0, max_real=50_000, y_real=None, y_syn=None):
    """Out-of-fold p(real) for every synthetic row + C2ST AUC. If labels are given, the real sample is
    class-matched to the synthetic set."""
    import xgboost as xgb

    rng = np.random.default_rng(seed)
    if y_real is not None and y_syn is not None:
        R = class_matched_real(Z_real, y_real, y_syn, seed=seed, max_real=max_real)
    else:
        n = min(len(Z_real), max(len(Z_syn), 1000), max_real)
        R = Z_real[rng.choice(len(Z_real), n, replace=False)]
    X = np.vstack([R, Z_syn]); y = np.r_[np.ones(len(R)), np.zeros(len(Z_syn))]
    w = np.where(y == 1, len(Z_syn) / max(len(R), 1), 1.0)
    p = np.zeros(len(X))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        m = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, tree_method="hist",
                              n_jobs=-1, random_state=seed, verbosity=0)
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y, p, sample_weight=w))
    return p[len(R):], auc


def dcr(Z_query, Z_ref, k=1):
    if len(Z_ref) == 0 or len(Z_query) == 0:
        return np.full(len(Z_query), np.inf)
    nn_ = NearestNeighbors(n_neighbors=min(k, len(Z_ref))).fit(Z_ref)
    d, _ = nn_.kneighbors(Z_query)
    return d[:, -1]


class TripleFilter:
    def __init__(self, rho=0.3, dcr_q=5.0, margin_hi_q=90.0, use_realism=True, use_dcr=True,
                 use_margin=True, min_keep_frac=0.1, seed=0):
        self.rho, self.dcr_q, self.margin_hi_q = rho, dcr_q, margin_hi_q
        self.min_keep_frac = min_keep_frac
        self.use = dict(realism=use_realism, dcr=use_dcr, margin=use_margin)
        self.seed = seed

    def __call__(self, Z_syn, y_syn, Z_tr, y_tr, Z_va, y_va, boundary, delta):
        keep = np.ones(len(Z_syn), bool)
        rep = {"n_in": int(len(Z_syn))}
        if self.use["realism"] and len(Z_syn):
            p, auc = realism_scores(Z_tr, Z_syn, seed=self.seed, y_real=y_tr, y_syn=y_syn)
            k = np.ones(len(Z_syn), bool)
            for c in np.unique(y_syn):
                ix = np.where(y_syn == c)[0]
                k[ix] = p[ix] >= np.quantile(p[ix], self.rho)
            rep.update(realism_keep=float(k.mean()), c2st_auc_prefilter=auc)
            keep &= k
        m = boundary.margin(Z_syn, y_syn) if boundary is not None else np.zeros(len(Z_syn))
        m_tr = boundary.margin(Z_tr, y_tr) if boundary is not None else None
        for c in np.unique(y_syn):
            ix = np.where(y_syn == c)[0]
            ref_tr = Z_tr[y_tr == c]
            if self.use["dcr"]:
                d_syn = dcr(Z_syn[ix], ref_tr)
                va = Z_va[y_va == c] if Z_va is not None else np.zeros((0, Z_tr.shape[1]))
                d_ref = dcr(va, ref_tr) if len(va) >= 10 else dcr(ref_tr, ref_tr, k=2)
                thr = np.percentile(d_ref, self.dcr_q)
                kd = d_syn >= thr
                rep[f"dcr_keep[{c}]"] = float(kd.mean())
                keep[ix] &= kd
            if self.use["margin"] and boundary is not None:
                hi = np.percentile(m_tr[y_tr == c], self.margin_hi_q)
                km = (m[ix] > 0) & (m[ix] <= hi)
                rep[f"margin_keep[{c}]"] = float(km.mean())
                if km.mean() < self.min_keep_frac:
                    # For a very rare class the ensemble itself is weak and almost no sample lands on
                    # the right side: dropping everything would deny augmentation to exactly the classes
                    # that need it. Relax to the top `min_keep_frac` by margin (highest m_y first).
                    thr = np.quantile(m[ix], 1.0 - self.min_keep_frac)
                    km = m[ix] >= thr
                    rep[f"margin_relaxed[{c}]"] = float(km.mean())
                keep[ix] &= km
        score = -np.abs(m - delta)
        rep["n_out"] = int(keep.sum())
        LOG.info("Filter: kept %d/%d samples", rep["n_out"], rep["n_in"])
        return keep, score, rep
