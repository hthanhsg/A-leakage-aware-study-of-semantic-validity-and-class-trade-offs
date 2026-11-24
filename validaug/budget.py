"""Per-class augmentation budget.

* ``default_budget`` (shared by ALL baselines for a fair comparison):
      target_c = min(max(N_c, alpha*N_max), beta*N_c) ; n_c = target_c - N_c ; majority class n = 0.
* ``AdaptiveBudget`` (VALID-Aug): per-class ratio r_c on a coarse geometric grid
  {0, 0.25, 0.5, 1, 2, 5, beta} (Optuna TPE over categorical choices), objective = val macro-F1
  (leakage-free split) of a proxy ensemble trained on train + the top-(r_c*N_c) filtered samples.
  The number of synthetic samples added for class c is  n_c = min(r_c * N_c, |pool_c|) <= beta * N_c.

Design notes (why not a continuous search in [0, beta]):
  * with ~10 classes and 30 trials, a continuous TPE search is far too sparse and most random
    trials over-augment, so the enqueued "no augmentation" point wins every tie -> the method
    silently degenerates to the baseline;
  * the proxy is an ensemble (XGBoost + small MLP by default): augmentation mostly helps neural
    classifiers, a tree-only proxy would pick r_c = 0 even where MLP/CNN gain a lot;
  * a trial must beat "no augmentation" by at least ``min_gain`` (noise floor of the proxy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .utils import LOG

GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


def class_counts(y):
    return pd.Series(y).value_counts().to_dict()


def default_budget(y_train, alpha=0.5, beta=10.0) -> dict:
    cnt = class_counts(y_train)
    nmax = max(cnt.values())
    out = {}
    for c, n in cnt.items():
        if n == nmax:
            out[c] = 0
            continue
        target = min(max(n, alpha * nmax), beta * n)
        out[c] = int(max(0, round(target - n)))
    return out


def restrict_val(Z_va, y_va, classes):
    """Keep only validation rows whose class exists in the training classes."""
    ok = np.isin(np.asarray(y_va, dtype=object), list(classes))
    return Z_va[ok], np.asarray(y_va, dtype=object)[ok]


def stratified_cap(Z, y, cap, seed):
    """Cap a set to `cap` rows keeping every class (water-filling, rare classes kept whole)."""
    if len(Z) <= cap:
        return Z, y
    from .runner import cap_train

    ix = cap_train(np.arange(len(Z)), np.asarray(y, dtype=object), cap, seed)
    return Z[ix], np.asarray(y, dtype=object)[ix]


def _proxy_f1(Ztr, ytr, Zva, yva, classes, seed, proxies=("xgboost",), device="cpu"):
    """Mean val macro-F1 of the proxy classifiers (fast settings)."""
    from .classifiers import make_classifier

    Zva, yva = restrict_val(Zva, yva, classes)
    cid = {c: i for i, c in enumerate(classes)}
    yv = np.array([cid[v] for v in yva])
    fast = {"xgboost": dict(n_estimators=120, max_depth=6, learning_rate=0.15, early_stopping_rounds=None),
            "mlp": dict(epochs=8, patience=3, min_epochs=3, batch_size=1024),
            "rf": dict(n_estimators=100)}
    scores = []
    for name in proxies:
        m = make_classifier(name, classes, seed, device, fast.get(name, {})).fit(Ztr, ytr, Zva, yva)
        pred = m.predict_proba(Zva).argmax(1)
        scores.append(f1_score(yv, pred, average="macro", labels=list(range(len(classes))), zero_division=0))
    return float(np.mean(scores))


class AdaptiveBudget:
    def __init__(self, beta=10.0, n_trials=30, seed=0, max_val=50_000, proxies=("xgboost", "mlp"),
                 min_gain=0.002, grid=GRID, device="cpu"):
        self.beta, self.n_trials, self.seed, self.max_val = beta, n_trials, seed, max_val
        self.proxies, self.min_gain, self.device = tuple(proxies), min_gain, device
        self.grid = tuple(sorted({float(g) for g in grid if g <= beta} | {0.0, float(beta)}))

    def search(self, Z_tr, y_tr, Z_va, y_va, pools: dict) -> tuple[dict, dict]:
        """pools: {cls: Z_pool} sorted by decreasing filter score."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cnt = class_counts(y_tr)
        classes = sorted(cnt)
        Z_va, y_va = restrict_val(Z_va, y_va, classes)
        Z_va, y_va = stratified_cap(Z_va, y_va, self.max_val, self.seed)
        minority = [c for c in pools if len(pools[c]) > 0]
        grid = list(self.grid)

        def build(rs):
            Zs, ys = [Z_tr], [y_tr]
            n_sel = {}
            for c in minority:
                k = int(min(len(pools[c]), round(rs[c] * cnt[c])))
                n_sel[c] = k
                if k:
                    Zs.append(pools[c][:k]); ys.append(np.full(k, c, dtype=object))
            return np.vstack(Zs), np.concatenate(ys), n_sel

        def score(rs):
            Z, y, _ = build(rs)
            return _proxy_f1(Z, y, Z_va, y_va, classes, self.seed, self.proxies, self.device)

        def objective(trial):
            rs = {c: trial.suggest_categorical(f"r[{c}]", grid) for c in minority}
            return score(rs)

        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed))
        st.enqueue_trial({f"r[{c}]": 0.0 for c in minority})          # no augmentation
        st.enqueue_trial({f"r[{c}]": 1.0 for c in minority})          # double every minority class
        st.enqueue_trial({f"r[{c}]": grid[-1] for c in minority})     # full upper bound (= beta)
        st.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        f1_none = st.trials[0].value
        # best trial that beats "no augmentation" by the noise floor; else fall back to r = 0
        cand = [t for t in st.trials if t.value is not None and t.value >= f1_none + self.min_gain]
        best_t = max(cand, key=lambda t: t.value) if cand else st.trials[0]
        best = {c: float(best_t.params[f"r[{c}]"]) for c in minority}
        _, _, n_sel = build(best)
        info = {"best_val_f1": float(best_t.value), "val_f1_no_aug": float(f1_none),
                "gain": float(best_t.value - f1_none), "ratios": best, "n_selected": n_sel,
                "proxies": list(self.proxies), "grid": grid, "n_trials": len(st.trials),
                "fell_back_to_zero": bool(not cand),
                "trials": [{"r": {k[2:-1]: v for k, v in t.params.items()}, "val_f1": t.value} for t in st.trials]}
        LOG.info("Adaptive budget: val F1 %.4f (no augmentation %.4f, gain %+.4f), selected %s%s",
                 best_t.value, f1_none, best_t.value - f1_none, n_sel,
                 "  [no trial beat the noise floor -> r=0]" if not cand else "")
        return n_sel, info
