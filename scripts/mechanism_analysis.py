"""Mechanism analyses for the evaluation study (CPU only; reads processed data, saved splits and results).

Three stages, each answering one "why" question behind the main tables:

  ceiling : How much of the test error is irreducible?  Flows with identical feature vectors but
            different labels cannot all be classified correctly by ANY model.  Reports the exact
            accuracy ceiling, an upper bound and an achievable reference for macro-F1, per-class
            F1 ceilings, the share of test flows that contradict a training flow, and (for the
            naive protocol) how far a pure exact-match lookup table gets.
  shift   : How different is test from train under each splitting protocol?  Domain-classifier
            AUC (overall and class-conditional), mean per-feature Wasserstein distance and the
            exact-match rate, related to the score of the un-augmented classifier.
  overlap : Which classes are hurt by augmentation?  Prior-free k-NN class overlap (k-disagreeing
            neighbours on a class-balanced training subsample) and "exposure" to the synthetic mass
            of neighbouring classes, related to the per-class F1 change of every augmenter.

Nothing here touches the experiment runs: splits are loaded read-only from <out_dir>/splits of the
given config (or rebuilt deterministically), and all outputs go to --out.

  python scripts/mechanism_analysis.py --exp configs/phaseB_protocol.yaml \
      --results runs/phaseB_protocol/results.jsonl runs/phaseD_main/results.jsonl \
      --out runs/mechanism --seeds 0 1 2 --n-jobs 8
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validaug.encoder import TabularEncoder  # noqa: E402
from validaug.io import feature_columns  # noqa: E402
from validaug.splits import make_split  # noqa: E402
from validaug.utils import LOG, dump_json, setup_logging  # noqa: E402

BENIGN = "Benign"
DS_SHORT = {"nf_unsw_v3": "UNSW", "nf_ton_v3": "ToN", "nf_bot_v3": "BoT", "nf_cic2018_v3": "CIC18",
            "cic2017_fixed": "CIC17"}
SPLIT_SHORT = {"random_naive": "Naive", "random": "Random", "temporal_stratified": "Temporal", "group": "Group",
               "temporal_global": "Temp.-global"}


# ----------------------------------------------------------------------------------- data access
def load_dataset(data_dir, name):
    d = Path(data_dir) / name
    df = pd.read_parquet(d / "data.parquet")
    rep = json.load(open(d / "audit.json"))
    return df, rep


def col_types(df, rep):
    cat = [c for c in rep["categorical"] if c in df.columns]
    bit = [c for c in rep["bit_features"] if c in df.columns]
    num = [c for c in feature_columns(df) if c not in cat + bit]
    return num, cat, bit


def get_split(cfg, df, ds, split, seed):
    """Same rule as Runner._split, but read-only: load the saved indices if present, else rebuild."""
    sseed = 0 if cfg.get("split_seed_fixed", False) else seed
    p = Path(cfg["out_dir"]) / "splits" / f"{ds}__{split}__{sseed}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        return {"train": z["train"], "val": z["val"], "test": z["test"]}, "saved"
    sp = make_split(df, split, sseed, train_conflicts=cfg.get("train_conflicts", "majority"))
    return sp, "rebuilt"


def subsample(idx, n, rng):
    return idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))


def load_results(paths):
    rows = []
    for p in paths:
        if not Path(p).exists():
            LOG.warning("results file not found: %s", p)
            continue
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates(["dataset", "split", "scenario", "seed", "method", "classifier"], keep="last")
    return df


# ----------------------------------------------------------------------------------- stage 1: ceiling
def best_f1_for_class(pos, tot):
    """Largest F1 any deterministic function of the features can reach for one class.

    pos[g], tot[g]: number of class-c flows / all flows in feature group g.  A feature-based classifier
    predicts the same label for a whole group, so it chooses a SET of groups to call "c".  For a
    fractional-linear objective the optimal set is a prefix of the groups sorted by purity pos/tot."""
    P = pos.sum()
    if P == 0:
        return np.nan
    keep = pos > 0
    pos, tot = pos[keep], tot[keep]
    o = np.argsort(-(pos / tot), kind="stable")
    tp, pp = np.cumsum(pos[o]), np.cumsum(tot[o])
    return float(np.max(2 * tp / (pp + P)))


def ceiling_cell(h, y, tr, te):
    """h: feature-only row hash of the whole dataset; y: labels; tr/te: row indices."""
    yt = y[te]
    d = pd.DataFrame({"h": h[te], "y": yt})
    ct = d.groupby(["h", "y"], sort=False).size().rename("n").reset_index()
    tot = ct.groupby("h")["n"].transform("sum").to_numpy()
    ct["tot"] = tot
    # (a) exact accuracy ceiling + achievable reference: predict the majority label of each group
    top = ct.sort_values("n", ascending=False).drop_duplicates("h")
    maj = dict(zip(top["h"].to_numpy(), top["y"].to_numpy()))
    pred = d["h"].map(maj).to_numpy(object)
    classes = sorted(set(yt))
    out = {"n_test": int(len(te)), "n_test_classes": len(classes),
           "acc_ceiling": float((pred == yt).mean()),
           "macro_f1_majority_oracle": float(f1_score(yt, pred, average="macro", labels=classes, zero_division=0)),
           "mcc_majority_oracle": float(matthews_corrcoef(yt, pred)),
           "test_ambiguous_frac": float((d.groupby("h")["y"].transform("nunique") > 1).mean())}
    # (b) upper bound: each class optimised on its own (jointly unattainable in general)
    per = {}
    for c, g in ct.groupby("y", sort=False):
        per[c] = best_f1_for_class(g["n"].to_numpy(float), g["tot"].to_numpy(float))
    out["macro_f1_upper_bound"] = float(np.nanmean(list(per.values())))
    f1m = f1_score(yt, pred, average=None, labels=classes, zero_division=0)
    # (c) relation to the training partition
    dtr = pd.DataFrame({"h": h[tr], "y": y[tr]})
    trc = dtr.groupby(["h", "y"], sort=False).size().rename("n").reset_index()
    trmaj = trc.sort_values("n", ascending=False).drop_duplicates("h")
    trmap = dict(zip(trmaj["h"].to_numpy(), trmaj["y"].to_numpy()))
    seen = d["h"].isin(trmap.keys()).to_numpy()
    look = d["h"].map(trmap).to_numpy(object)
    contra = seen & (look != yt)
    out.update(test_seen_in_train_frac=float(seen.mean()),
               lookup_acc_on_seen=float((look[seen] == yt[seen]).mean()) if seen.any() else np.nan,
               test_contradicts_train_frac=float(contra.mean()))
    rows = []
    amb = (d.groupby("h")["y"].transform("nunique") > 1).to_numpy()
    for c, f in zip(classes, f1m):
        m = yt == c
        rows.append({"class": c, "n_test": int(m.sum()), "f1_upper_bound": per.get(c, np.nan),
                     "f1_majority_oracle": float(f), "ambiguous_frac": float(amb[m].mean()),
                     "seen_in_train_frac": float(seen[m].mean()), "contradicts_train_frac": float(contra[m].mean())})
    return out, rows


# ----------------------------------------------------------------------------------- stage 2: shift
def _c2st(Za, Zb, seed, n_jobs):
    """Domain-classifier AUC between two samples (0.5 = indistinguishable, 1 = fully separable)."""
    from xgboost import XGBClassifier

    X = np.vstack([Za, Zb]).astype(np.float32)
    y = np.r_[np.zeros(len(Za), int), np.ones(len(Zb), int)]
    if min(len(Za), len(Zb)) < 20:
        return np.nan
    p = np.zeros(len(y))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y):
        m = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.2, subsample=0.8, colsample_bytree=0.8,
                          tree_method="hist", n_jobs=n_jobs, random_state=seed, verbosity=0)
        m.fit(X[a], y[a])
        p[b] = m.predict_proba(X[b])[:, 1]
    return float(roc_auc_score(y, p))


def shift_cell(Ztr, ytr, Zte, yte, n_num, seed, n_jobs, n_all=20000, n_cls=4000, min_cls=100):
    rng = np.random.default_rng(seed)
    out = {}
    # (a) class-balanced marginal shift: both sides resampled to the same class mix, so the number is
    #     not driven by different class priors of train and test
    common = sorted(set(ytr) & set(yte))
    per_c = max(50, n_all // max(1, len(common)))
    ia = np.concatenate([subsample(np.where(ytr == c)[0], per_c, rng) for c in common])
    ib = np.concatenate([subsample(np.where(yte == c)[0], per_c, rng) for c in common])
    out["c2st_auc"] = _c2st(Ztr[ia], Zte[ib], seed, n_jobs)
    w = [wasserstein_distance(Ztr[ia, j], Zte[ib, j]) for j in range(n_num)]
    out["wasserstein_mean"] = float(np.mean(w)) if w else np.nan
    # (b) class-conditional shift: is a test flow of class c recognisably different from a train flow of c?
    rows = []
    for c in common:
        a, b = np.where(ytr == c)[0], np.where(yte == c)[0]
        if min(len(a), len(b)) < min_cls:
            continue
        n = min(len(a), len(b), n_cls)
        a, b = subsample(a, n, rng), subsample(b, n, rng)
        rows.append({"class": c, "n": int(n), "c2st_auc": _c2st(Ztr[a], Zte[b], seed, n_jobs)})
    v = [r["c2st_auc"] for r in rows if not np.isnan(r["c2st_auc"])]
    out["c2st_auc_classcond"] = float(np.mean(v)) if v else np.nan
    out["c2st_auc_classcond_max"] = float(np.max(v)) if v else np.nan
    out["n_classes_classcond"] = len(v)
    return out, rows


# ----------------------------------------------------------------------------------- stage 3: overlap
def overlap_cell(Ztr, ytr, seed, n_per_class=2000, k=10, n_jobs=1):
    """k-disagreeing-neighbours on a class-balanced subsample -> prior-free overlap.

    O[c, j] = mean share of the k nearest neighbours of a class-c flow that carry label j.
    overlap_c = 1 - O[c, c]."""
    rng = np.random.default_rng(seed)
    classes = sorted(set(ytr))
    idx = np.concatenate([subsample(np.where(ytr == c)[0], n_per_class, rng) for c in classes])
    Z, y = Ztr[idx].astype(np.float32), ytr[idx]
    # every class gets the same total weight even if it has fewer than n_per_class flows
    wcls = {c: 1.0 / max(1, int((y == c).sum())) for c in classes}
    kk = min(k + 1, len(Z))
    nn = NearestNeighbors(n_neighbors=kk, n_jobs=n_jobs).fit(Z)
    nb = nn.kneighbors(Z, return_distance=False)[:, 1:]
    cid = {c: i for i, c in enumerate(classes)}
    yi = np.array([cid[v] for v in y])
    wn = np.array([wcls[c] for c in classes])[yi]            # weight of each point as a neighbour
    O = np.zeros((len(classes), len(classes)))
    for i, c in enumerate(classes):
        m = yi == i
        nbl, nbw = yi[nb[m]].ravel(), wn[nb[m]].ravel()
        O[i] = np.bincount(nbl, weights=nbw, minlength=len(classes))
        O[i] /= max(O[i].sum(), 1e-12)
    return classes, O


def per_class_delta(res, split, scenario="full"):
    """F1 change of every class under every augmenter, paired with `none` on (dataset, seed, classifier)."""
    if not len(res):
        return pd.DataFrame()
    r = res[(res["split"] == split) & (res["scenario"] == scenario) & res["f1_per_class"].notna()]
    base = {(a.dataset, a.seed, a.classifier): a.f1_per_class for a in r[r.method == "none"].itertuples()}
    rows = []
    for a in r[r.method != "none"].itertuples():
        b = base.get((a.dataset, a.seed, a.classifier))
        if b is None:
            continue
        syn = a.n_syn_per_class if isinstance(getattr(a, "n_syn_per_class", None), dict) else {}
        for c, f in a.f1_per_class.items():
            if c in b:
                rows.append({"dataset": a.dataset, "seed": a.seed, "classifier": a.classifier, "method": a.method,
                             "class": c, "f1_none": b[c], "delta_f1": f - b[c], "n_syn": float(syn.get(c, 0))})
    return pd.DataFrame(rows)


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def _boot_ci(x, y, groups, n=2000, seed=0):
    """Cluster bootstrap over datasets is too coarse (5 clusters); resample (dataset, class) units."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        r, _, _ = _spearman(x[i], y[i])
        if np.isfinite(r):
            vals.append(r)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) > 50 else (np.nan, np.nan)


def _center(df, col, by="dataset"):
    return df[col] - df.groupby(by)[col].transform("mean")


# ----------------------------------------------------------------------------------- output helpers
def _tex(df, path, caption, label, fmt="%.3f"):
    body = df.to_latex(index=False, escape=True, na_rep="--", float_format=lambda v: fmt % v)
    Path(path).write_text("\\begin{table}[t]\n\\centering\\small\n\\caption{%s}\n\\label{%s}\n%s\\end{table}\n"
                          % (caption, label, body))


def _figures(out, ceil, shift, octab, corr):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # figures are optional
        LOG.warning("matplotlib not available (%s) -> figures skipped", e)
        return
    INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
    HUE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    MARK = ["o", "s", "^", "D", "v"]
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "legend.frameon": False, "pdf.fonttype": 42})

    def ds_style(names):
        order = [d for d in DS_SHORT if d in names] + sorted(set(names) - set(DS_SHORT))
        return {d: (HUE[i % len(HUE)], MARK[i % len(MARK)]) for i, d in enumerate(order[:5])}

    if len(shift) and "none_macro_f1" in shift and shift["none_macro_f1"].notna().any():
        st = ds_style(list(shift["dataset"].unique()))
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        for d, g in shift.groupby("dataset"):
            if d not in st:
                continue
            g = g.sort_values("c2st_auc_classcond")
            ax.plot(g["c2st_auc_classcond"], g["none_macro_f1"], color=st[d][0], lw=1, alpha=0.5, zorder=1)
            ax.scatter(g["c2st_auc_classcond"], g["none_macro_f1"], color=st[d][0], marker=st[d][1], s=22,
                       edgecolor="white", linewidth=0.6, label=DS_SHORT.get(d, d), zorder=2)
        ax.set_xlabel("Class-conditional train-test C2ST AUC")
        ax.set_ylabel("Macro-F1 without augmentation")
        ax.legend(fontsize=7, ncol=2, loc="best")
        fig.tight_layout(); fig.savefig(out / "fig_shift_vs_f1.pdf"); plt.close(fig)

    if len(ceil) and "none_macro_f1_best" in ceil:
        c = ceil[ceil["none_macro_f1_best"].notna()]
        splits = [s for s in SPLIT_SHORT if s in set(c["split"])]
        if len(c) and splits:
            fig, axes = plt.subplots(1, len(splits), figsize=(1.75 * len(splits), 2.4), sharey=True, squeeze=False)
            for ax, s in zip(axes[0], splits):
                g = c[c["split"] == s].set_index("dataset")
                g = g.loc[[d for d in DS_SHORT if d in g.index] + [d for d in g.index if d not in DS_SHORT]]
                x = np.arange(len(g))
                ax.bar(x, g["macro_f1_upper_bound"], width=0.62, color="#cde2fb", label="Upper bound (label conflicts)")
                ax.bar(x, g["none_macro_f1_best"], width=0.36, color="#2a78d6", label="Best classifier, no augmentation")
                ax.set_xticks(x); ax.set_xticklabels([DS_SHORT.get(d, d) for d in g.index], rotation=45, ha="right")
                ax.set_title(SPLIT_SHORT[s], fontsize=8, color=INK); ax.grid(axis="x", visible=False)
            axes[0][0].set_ylabel("Macro-F1"); axes[0][0].set_ylim(0, 1.02)
            h, l = axes[0][0].get_legend_handles_labels()
            fig.legend(h, l, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.02))
            fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out / "fig_ceiling.pdf"); plt.close(fig)

    if len(octab) and "delta_f1" in octab:
        show = [m for m in ("smote", "ctgan", "diffusion", "validaug") if m in set(octab["method"])][:4]
        if show:
            st = ds_style(list(octab["dataset"].unique()))
            fig, axes = plt.subplots(1, len(show), figsize=(max(3.4, 1.8 * len(show)), 2.5), sharey=True, squeeze=False)
            for ax, m in zip(axes[0], show):
                g = octab[octab["method"] == m]
                for d, gg in g.groupby("dataset"):
                    if d in st:
                        ax.scatter(gg["overlap"], gg["delta_f1"], color=st[d][0], marker=st[d][1], s=16,
                                   edgecolor="white", linewidth=0.5, label=DS_SHORT.get(d, d))
                ax.axhline(0, color=MUTED, lw=0.8)
                rr = corr[(corr["method"] == m) & (corr["x"] == "overlap") & (corr["variant"] == "pooled")]
                t = m if not len(rr) else "%s ($\\rho$=%.2f)" % (m, rr["rho"].iloc[0])
                ax.set_title(t, fontsize=8, color=INK); ax.set_xlabel("Class overlap")
            axes[0][0].set_ylabel("$\\Delta$F1 of the class vs. no augmentation")
            h, l = axes[0][0].get_legend_handles_labels()
            fig.legend(h, l, loc="upper center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 1.02))
            fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out / "fig_overlap_vs_delta.pdf"); plt.close(fig)


# ----------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="experiment config (data_dir, datasets, splits, seeds, out_dir)")
    ap.add_argument("--results", nargs="*", default=[], help="results.jsonl files (phase B and/or D)")
    ap.add_argument("--out", default="runs/mechanism")
    ap.add_argument("--stages", nargs="+", default=["ceiling", "shift", "overlap"],
                    choices=["ceiling", "shift", "overlap"])
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--splits", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--overlap-split", default="temporal_stratified",
                    help="protocol whose per-class results are explained by the overlap stage")
    ap.add_argument("--overlap-per-class", type=int, default=2000)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.exp))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.get("log_level", "INFO"), str(out / "mechanism.log"))
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import logging
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    datasets = a.datasets or cfg["datasets"]
    splits = a.splits or cfg["splits"]
    seeds = a.seeds if a.seeds is not None else cfg["seeds"]
    res = load_results(a.results)
    LOG.info("stages=%s datasets=%s splits=%s seeds=%s ; %d result rows", a.stages, datasets, splits, seeds, len(res))
    cap = cfg.get("max_train_rows")

    ceil_rows, ceil_cls, shift_rows, shift_cls, ov_rows, ov_pairs = [], [], [], [], [], []
    t_all = time.time()
    for ds in datasets:
        t0 = time.time()
        df, rep = load_dataset(cfg["data_dir"], ds)
        y = df["__y"].to_numpy(object)
        feats = feature_columns(df)
        h = pd.util.hash_pandas_object(df[feats], index=False).to_numpy()
        num, cat, bit = col_types(df, rep)
        LOG.info("[%s] %d rows, %d features, %d classes (loaded in %.0fs)", ds, len(df), len(feats), len(set(y)),
                 time.time() - t0)
        for split in splits:
            if (split.startswith("temporal") and not rep.get("has_time", True)) or \
                    (split == "group" and not rep.get("has_host", True)):
                LOG.warning("[%s] split %s not available -> skipped", ds, split)
                continue
            for seed in seeds:
                t1 = time.time()
                sp, src = get_split(cfg, df, ds, split, seed)
                tr, te = np.asarray(sp["train"]), np.asarray(sp["test"])
                key = {"dataset": ds, "split": split, "seed": seed}
                if "ceiling" in a.stages:
                    o, rows = ceiling_cell(h, y, tr, te)
                    ceil_rows.append({**key, **o})
                    ceil_cls += [{**key, **r} for r in rows]
                need_enc = "shift" in a.stages or ("overlap" in a.stages and split == a.overlap_split)
                if need_enc:
                    rng = np.random.default_rng(seed)
                    trs = tr
                    if cap and len(trs) > cap:
                        trs = subsample(trs, cap, rng)
                    # the encoder is fitted on (a sample of) train only, exactly like in the experiments
                    fit_idx = subsample(trs, 200000, rng)
                    enc = TabularEncoder(**cfg.get("encoder", {})).fit(df.iloc[fit_idx], num, cat, bit)
                    keep_tr = np.concatenate([subsample(trs[y[trs] == c], 6000, rng) for c in sorted(set(y[trs]))])
                    keep_te = np.concatenate([subsample(te[y[te] == c], 6000, rng) for c in sorted(set(y[te]))])
                    Ztr = enc.transform(enc.clean(df.iloc[keep_tr].reset_index(drop=True)))
                    Zte = enc.transform(enc.clean(df.iloc[keep_te].reset_index(drop=True)))
                    ytr, yte = y[keep_tr], y[keep_te]
                if "shift" in a.stages:
                    o, rows = shift_cell(Ztr, ytr, Zte, yte, len(enc.num_cols), seed, a.n_jobs)
                    shift_rows.append({**key, **o})
                    shift_cls += [{**key, **r} for r in rows]
                if "overlap" in a.stages and split == a.overlap_split:
                    classes, O = overlap_cell(Ztr, ytr, seed, a.overlap_per_class, a.knn, a.n_jobs)
                    ntr = pd.Series(y[tr]).value_counts()
                    for i, c in enumerate(classes):
                        j = int(np.argsort(-np.where(np.arange(len(classes)) == i, -1, O[i]))[0]) if len(classes) > 1 else i
                        ov_rows.append({**key, "class": c, "n_train": int(ntr.get(c, 0)), "overlap": float(1 - O[i, i]),
                                        "nearest_class": classes[j], "nearest_share": float(O[i, j])})
                        ov_pairs += [{**key, "class": c, "other": classes[q], "share": float(O[i, q])}
                                     for q in range(len(classes)) if q != i]
                LOG.info("[%s|%s|s%d] done in %.0fs (split %s)", ds, split, seed, time.time() - t1, src)
        del df

    summary = {"stages": a.stages, "datasets": datasets, "splits": splits, "seeds": seeds,
               "wall_s": None}
    none = res[(res["method"] == "none") & (res["scenario"] == "full")] if len(res) else pd.DataFrame()

    # ------------------------------------------------------------------ ceiling tables
    ceil = pd.DataFrame()
    if ceil_rows:
        pd.DataFrame(ceil_rows).to_csv(out / "ceiling_cells.csv", index=False)
        pd.DataFrame(ceil_cls).to_csv(out / "ceiling_per_class_cells.csv", index=False)
        ceil = pd.DataFrame(ceil_rows).groupby(["dataset", "split"], sort=False).mean(numeric_only=True) \
            .drop(columns=["seed"]).reset_index()
        if len(none):
            g = none.groupby(["dataset", "split", "classifier"])[["macro_f1", "accuracy", "mcc"]].mean().reset_index()
            best = g.sort_values("macro_f1", ascending=False).drop_duplicates(["dataset", "split"]) \
                .rename(columns={"macro_f1": "none_macro_f1_best", "accuracy": "none_acc_best", "mcc": "none_mcc_best",
                                 "classifier": "best_classifier"})
            ceil = ceil.merge(best, on=["dataset", "split"], how="left")
            ceil["gap_to_upper_bound"] = ceil["macro_f1_upper_bound"] - ceil["none_macro_f1_best"]
            ceil["gap_to_majority_oracle"] = ceil["macro_f1_majority_oracle"] - ceil["none_macro_f1_best"]
            ceil["acc_gap"] = ceil["acc_ceiling"] - ceil["none_acc_best"]
        ceil.to_csv(out / "ceiling.csv", index=False)
        pc = pd.DataFrame(ceil_cls).groupby(["dataset", "split", "class"], sort=False).mean(numeric_only=True) \
            .drop(columns=["seed"]).reset_index()
        if len(none):
            f = [{"dataset": r.dataset, "split": r.split, "class": c, "f1": v}
                 for r in none[none["f1_per_class"].notna()].itertuples() for c, v in r.f1_per_class.items()]
            if f:
                pc = pc.merge(pd.DataFrame(f).groupby(["dataset", "split", "class"])["f1"].mean()
                              .rename("none_f1_mean").reset_index(), on=["dataset", "split", "class"], how="left")
        pc.to_csv(out / "ceiling_per_class.csv", index=False)
        cols = ["dataset", "split", "test_ambiguous_frac", "test_contradicts_train_frac", "test_seen_in_train_frac",
                "acc_ceiling", "macro_f1_majority_oracle", "macro_f1_upper_bound"] + \
               [c for c in ("none_macro_f1_best", "gap_to_upper_bound") if c in ceil]
        t = ceil[cols].copy()
        t["dataset"] = t["dataset"].map(lambda d: DS_SHORT.get(d, d)); t["split"] = t["split"].map(lambda s: SPLIT_SHORT.get(s, s))
        t.columns = ["Dataset", "Protocol", "Ambig.", "Contra.", "Seen", "Acc. ceil.", "F1 maj.", "F1 upper"] + \
                    (["F1 none", "Gap"] if "none_macro_f1_best" in ceil else [])
        _tex(t, out / "tab_ceiling.tex",
             "Label-conflict ceiling on the test partition. Ambig.: share of test flows whose feature vector occurs "
             "with more than one label in test; Contra.: share whose feature vector occurs in train with a different "
             "label; Seen: share whose feature vector occurs in train; Acc. ceil.: exact accuracy ceiling of any "
             "feature-based classifier; F1 maj./F1 upper: achievable reference and upper bound of macro-F1; "
             "F1 none: best classifier without augmentation.", "tab:ceiling")
        summary["ceiling"] = {"mean_gap_to_upper_bound_by_split":
                              ceil.groupby("split")["gap_to_upper_bound"].mean().to_dict() if "gap_to_upper_bound" in ceil else None,
                              "mean_macro_f1_upper_bound_by_split": ceil.groupby("split")["macro_f1_upper_bound"].mean().to_dict()}

    # ------------------------------------------------------------------ shift tables
    shift = pd.DataFrame()
    if shift_rows:
        pd.DataFrame(shift_rows).to_csv(out / "shift_cells.csv", index=False)
        pd.DataFrame(shift_cls).to_csv(out / "shift_per_class_cells.csv", index=False)
        shift = pd.DataFrame(shift_rows).groupby(["dataset", "split"], sort=False).mean(numeric_only=True) \
            .drop(columns=["seed"]).reset_index()
        if ceil_rows:
            shift = shift.merge(ceil[["dataset", "split", "test_seen_in_train_frac"]], on=["dataset", "split"], how="left")
        if len(none):
            m = none.groupby(["dataset", "split"])[["macro_f1", "mcc"]].mean().reset_index() \
                .rename(columns={"macro_f1": "none_macro_f1", "mcc": "none_mcc"})
            shift = shift.merge(m, on=["dataset", "split"], how="left")
            corr = []
            for x in ("c2st_auc", "c2st_auc_classcond", "wasserstein_mean", "test_seen_in_train_frac"):
                if x not in shift:
                    continue
                for ycol in ("none_macro_f1", "none_mcc"):
                    r, p, n = _spearman(shift[x], shift[ycol])
                    rc, pc_, _ = _spearman(_center(shift, x), _center(shift, ycol))
                    corr.append({"x": x, "y": ycol, "n_cells": n, "rho_pooled": r, "p_pooled": p,
                                 "rho_within_dataset": rc, "p_within_dataset": pc_})
            pd.DataFrame(corr).to_csv(out / "shift_correlation.csv", index=False)
            summary["shift_correlation"] = corr
        shift.to_csv(out / "shift.csv", index=False)
        cols = ["dataset", "split", "c2st_auc", "c2st_auc_classcond", "wasserstein_mean"] + \
               [c for c in ("test_seen_in_train_frac", "none_macro_f1") if c in shift]
        t = shift[cols].copy()
        t["dataset"] = t["dataset"].map(lambda d: DS_SHORT.get(d, d)); t["split"] = t["split"].map(lambda s: SPLIT_SHORT.get(s, s))
        t.columns = ["Dataset", "Protocol", "C2ST", "C2ST class-cond.", "Wasserstein"] + \
                    [n for c, n in (("test_seen_in_train_frac", "Seen"), ("none_macro_f1", "F1 none")) if c in shift]
        _tex(t, out / "tab_shift.tex",
             "Train-to-test distribution shift per splitting protocol. C2ST: AUC of a classifier separating train "
             "from test flows (class-balanced; 0.5 = indistinguishable); class-cond.: mean over classes of the same "
             "AUC computed within a class; Wasserstein: mean per-feature distance in the encoded space.", "tab:shift")

    # ------------------------------------------------------------------ overlap tables
    octab, ocorr = pd.DataFrame(), pd.DataFrame()
    if ov_rows:
        ov = pd.DataFrame(ov_rows)
        ov.to_csv(out / "overlap_cells.csv", index=False)
        pairs = pd.DataFrame(ov_pairs).groupby(["dataset", "class", "other"], sort=False)["share"].mean().reset_index()
        pairs.to_csv(out / "overlap_pairs.csv", index=False)
        ovm = ov.groupby(["dataset", "class"], sort=False).agg(
            overlap=("overlap", "mean"), n_train=("n_train", "mean"),
            nearest_class=("nearest_class", lambda s: s.mode().iloc[0]), nearest_share=("nearest_share", "mean")).reset_index()
        ovm.to_csv(out / "overlap.csv", index=False)
        d = per_class_delta(res, a.overlap_split)
        if len(d):
            dm = d.groupby(["dataset", "method", "class"], sort=False).agg(
                delta_f1=("delta_f1", "mean"), f1_none=("f1_none", "mean"), n_syn=("n_syn", "mean")).reset_index()
            octab = dm.merge(ovm, on=["dataset", "class"], how="inner")
            octab["syn_ratio"] = octab["n_syn"] / octab["n_train"].clip(lower=1)
            octab["log_n_train"] = np.log10(octab["n_train"].clip(lower=1))
            # exposure: synthetic mass injected into the classes this class overlaps with
            sr = octab.set_index(["dataset", "method", "class"])["syn_ratio"].to_dict()
            expo = {}
            for (dsn, c), g in pairs.groupby(["dataset", "class"]):
                for m in octab["method"].unique():
                    expo[(dsn, m, c)] = float(sum(s * np.log1p(sr.get((dsn, m, o), 0.0))
                                                  for o, s in zip(g["other"], g["share"])))
            octab["exposure"] = [expo.get((r.dataset, r.method, r["class"]), np.nan) for _, r in octab.iterrows()]
            octab.to_csv(out / "overlap_vs_delta.csv", index=False)
            rows = []
            for m, g in octab.groupby("method"):
                for x in ("overlap", "exposure", "syn_ratio", "log_n_train", "f1_none"):
                    r, p, n = _spearman(g[x], g["delta_f1"])
                    lo, hi = _boot_ci(g[x].to_numpy(), g["delta_f1"].to_numpy(), None)
                    rows.append({"method": m, "x": x, "variant": "pooled", "n": n, "rho": r, "p": p, "ci_lo": lo, "ci_hi": hi})
                    r, p, n = _spearman(_center(g, x), _center(g, "delta_f1"))
                    rows.append({"method": m, "x": x, "variant": "within_dataset", "n": n, "rho": r, "p": p,
                                 "ci_lo": np.nan, "ci_hi": np.nan})
                att = g[g["class"] != BENIGN]
                hi_ = att[att["overlap"] >= att["overlap"].median()]["delta_f1"]
                lo_ = att[att["overlap"] < att["overlap"].median()]["delta_f1"]
                rows.append({"method": m, "x": "overlap", "variant": "mean_delta_high_minus_low", "n": len(att),
                             "rho": float(hi_.mean() - lo_.mean()) if len(hi_) and len(lo_) else np.nan, "p": np.nan,
                             "ci_lo": float(lo_.mean()) if len(lo_) else np.nan, "ci_hi": float(hi_.mean()) if len(hi_) else np.nan})
            ocorr = pd.DataFrame(rows)
            ocorr.to_csv(out / "overlap_correlation.csv", index=False)
            p_ = ocorr[ocorr["variant"] == "pooled"].pivot(index="method", columns="x", values="rho")
            p_ = p_.reindex(columns=[c for c in ("overlap", "exposure", "syn_ratio", "log_n_train", "f1_none") if c in p_])
            hl = ocorr[ocorr["variant"] == "mean_delta_high_minus_low"].set_index("method")
            p_["dF1 low-overlap"], p_["dF1 high-overlap"] = hl["ci_lo"], hl["ci_hi"]
            p_ = p_.reset_index().rename(columns={"method": "Method", "overlap": "rho overlap", "exposure": "rho exposure",
                                                  "syn_ratio": "rho syn. ratio", "log_n_train": "rho log N",
                                                  "f1_none": "rho F1 none"})
            _tex(p_, out / "tab_overlap.tex",
                 "Spearman correlation between the per-class F1 change under augmentation and class properties "
                 "(units: dataset--class pairs, temporal protocol). Overlap: share of the k nearest neighbours of a "
                 "class that belong to other classes (class-balanced sample); exposure: overlap-weighted synthetic "
                 "mass added to neighbouring classes. Last columns: mean F1 change of attack classes below/above the "
                 "median overlap.", "tab:overlap", fmt="%.2f")
            summary["overlap_correlation_pooled"] = ocorr[ocorr["variant"] == "pooled"].to_dict("records")
        else:
            LOG.warning("no per-class results for split=%s scenario=full -> overlap written without correlations",
                        a.overlap_split)

    _figures(out, ceil, shift, octab, ocorr)
    summary["wall_s"] = round(time.time() - t_all, 1)
    dump_json(summary, out / "summary.json")
    LOG.info("finished in %.0fs -> %s", summary["wall_s"], out)


if __name__ == "__main__":
    main()
