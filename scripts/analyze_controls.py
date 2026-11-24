"""Analysis of the phase-G control experiments (CPU only; reads results.jsonl files, writes tables).

  cw     : cost-sensitive baselines (none_cw, none_cw_sqrt) vs none / oversamplers / VALID-Aug   (phase G1 + D)
  d50    : diffusion trained on the CTGAN/TVAE subsample size vs diffusion / CTGAN / TVAE       (phase G2 + D)
  nocap  : natural class imbalance (no per-class cap, no training cap)                          (phase G3)
  c2st   : why are constraint-valid synthetic flows still separable from real flows? Feature-level
           diagnosis of the real-vs-synthetic discriminator on the synthetic samples saved by phase D.

  python scripts/analyze_controls.py --main runs/phaseD_main/results.jsonl \
      --cw runs/phaseG_classweights/results.jsonl --d50 runs/phaseG_diff50k/results.jsonl \
      --nocap runs/phaseG_nocap/results.jsonl --c2st-exp configs/phaseD_main.yaml --out runs/phaseG_analysis
Every block is optional: a missing file is skipped.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validaug.utils import LOG, dump_json, setup_logging  # noqa: E402

KEY = ["dataset", "split", "scenario", "seed", "method", "classifier"]
METRICS = ["macro_f1", "rare_f1", "mcc"]


def load(path):
    if not path or not Path(path).exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    return df.drop_duplicates(KEY, keep="last") if len(df) else df


def holm(p):
    p = pd.Series(p, dtype=float).dropna()
    o = p.sort_values()
    m = len(o)
    return (o * (m - np.arange(m))).cummax().clip(upper=1.0)


def paired(g, ref, others, metric, unit=("dataset", "classifier")):
    """Paired comparison `ref - other` on blocks (default: dataset x classifier, averaged over seeds)."""
    P = g.groupby(list(unit) + ["method"])[metric].mean().unstack()
    rows = []
    for m in others:
        if m == ref or m not in P or ref not in P:
            continue
        d = (P[ref] - P[m]).dropna()
        if not len(d):
            continue
        p = wilcoxon(d).pvalue if (d != 0).any() and len(d) >= 5 else np.nan
        rows.append({"ref": ref, "vs": m, "metric": metric, "n_blocks": len(d), "mean_diff": d.mean(),
                     "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"]).reindex(out.index)
    return out


def common_cells(g, methods):
    """Keep only (dataset, seed, classifier) cells in which ALL the given methods are present."""
    have = g[g.method.isin(methods)].groupby(["dataset", "seed", "classifier"]).method.nunique()
    ok = have[have == len(set(methods) & set(g.method))].index
    return g.set_index(["dataset", "seed", "classifier"]).loc[lambda d: d.index.isin(ok)].reset_index()


def mean_table(g, methods):
    t = g.groupby(["method", "dataset"])[METRICS].mean().groupby("method").mean()
    return t.reindex([m for m in methods if m in t.index])


def to_tex(df, path, caption, label, dec=3):
    body = df.to_latex(escape=True, na_rep="--", float_format=lambda v: f"{v:.{dec}f}")
    Path(path).write_text("\\begin{table}[t]\n\\centering\\small\n\\caption{%s}\n\\label{%s}\n%s\\end{table}\n"
                          % (caption, label, body))


# ------------------------------------------------------------------------------------------- cw
def block_cw(main, cw, out, summary):
    if not len(cw):
        return
    allr = pd.concat([main, cw], ignore_index=True)
    allr = allr[allr.split == "temporal_stratified"]
    show = ["none", "none_cw", "none_cw_sqrt", "ros", "smote", "smote_enn", "diffusion", "validaug"]
    need = ["none", "none_cw", "none_cw_sqrt", "smote"]
    tabs, tests = [], []
    for scen in ("full", "rare5", "rare1"):
        g = allr[allr.scenario == scen]
        g = g[g.classifier.isin(set(cw[cw.scenario == scen].classifier))]
        g = common_cells(g, [m for m in need if m in set(g.method)])
        if not len(g):
            continue
        t = mean_table(g, show)
        t.insert(0, "n_cells", g.groupby("method").size().reindex(t.index))
        t.insert(0, "scenario", scen)
        tabs.append(t.reset_index())
        for cwm in ("none_cw", "none_cw_sqrt"):
            for met in METRICS:
                a = paired(g, cwm, ["none", "ros", "smote", "smote_enn", "validaug"], met)
                if len(a):
                    a.insert(0, "scenario", scen); tests.append(a)
        pc = g.groupby(["classifier", "method", "dataset"]).macro_f1.mean().groupby(["classifier", "method"]).mean() \
            .unstack().reindex(columns=[m for m in show if m in set(g.method)])
        pc.to_csv(out / f"cw_by_classifier__{scen}.csv")
    if not tabs:
        return
    T = pd.concat(tabs, ignore_index=True)
    T.to_csv(out / "cw_means.csv", index=False)
    S = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    S.to_csv(out / "cw_tests.csv", index=False)
    piv = T.pivot(index="method", columns="scenario", values="macro_f1").reindex([m for m in show if m in set(T.method)])
    to_tex(piv.reindex(columns=[c for c in ("full", "rare5", "rare1") if c in piv]), out / "tab_cw.tex",
           "Cost-sensitive learning versus augmentation (macro-F1, temporal protocol; cells and classifiers "
           "common to all listed methods). none\\_cw: balanced class weights; none\\_cw\\_sqrt: square-root weights.",
           "tab:cw")
    summary["cw"] = {"means": T.round(4).to_dict("records"),
                     "tests_macro_f1": S[S.metric == "macro_f1"].round(4).to_dict("records") if len(S) else []}
    LOG.info("class weights:\n%s", T.round(3).to_string(index=False))


# ------------------------------------------------------------------------------------------- d50
def block_d50(main, d50, out, summary):
    if not len(d50):
        return
    allr = pd.concat([main, d50], ignore_index=True)
    g = allr[(allr.split == "temporal_stratified") & (allr.scenario == "full")]
    g = g[g.classifier.isin(set(d50.classifier))]
    show = ["none", "ctgan", "tvae", "diffusion_50k", "diffusion", "validaug"]
    g = common_cells(g, [m for m in ("none", "ctgan", "tvae", "diffusion_50k", "diffusion") if m in set(g.method)])
    if not len(g):
        LOG.warning("d50: no common cells with phase D"); return
    byds = g.groupby(["method", "dataset"]).macro_f1.mean().unstack()
    byds["mean"] = byds.mean(axis=1)
    byds = byds.reindex([m for m in show if m in byds.index])
    byds.to_csv(out / "d50_macro_f1_by_dataset.csv")
    t = mean_table(g, show)
    ax = g.drop_duplicates(["dataset", "seed", "method"]).groupby("method")[
        [c for c in ("validity_raw", "c2st_auc", "wasserstein_mean", "aug_wall_s") if c in g]].mean()
    t = t.join(ax)
    t.to_csv(out / "d50_means.csv")
    tests = pd.concat([paired(g, "diffusion_50k", ["none", "ctgan", "tvae", "diffusion"], m) for m in METRICS],
                      ignore_index=True)
    tests.to_csv(out / "d50_tests.csv", index=False)
    to_tex(byds, out / "tab_d50.tex",
           "Generator training-set size (macro-F1, temporal protocol, full data). diffusion\\_50k is trained on the "
           "same class-capped 50,000-row subsample as CTGAN and TVAE.", "tab:d50")
    summary["d50"] = {"means": t.round(4).reset_index().to_dict("records"), "tests": tests.round(4).to_dict("records")}
    LOG.info("diffusion 50k:\n%s\n%s", byds.round(3).to_string(), tests.round(4).to_string(index=False))


# ------------------------------------------------------------------------------------------- nocap
def block_nocap(nocap, out, summary, data_dir):
    if not len(nocap):
        return
    g = nocap[(nocap.scenario == "full")]
    show = ["none", "none_cw", "none_cw_sqrt", "ros", "smote", "diffusion", "validaug"]
    seeds_by_method = g.groupby("method").seed.nunique().to_dict()
    g = common_cells(g, [m for m in show if m in set(g.method)])   # only seeds completed by ALL methods
    t = g.groupby(["dataset", "classifier", "method"])[METRICS].mean().unstack("method")
    t.to_csv(out / "nocap_by_dataset_classifier.csv")
    m = g.groupby(["dataset", "method"])[METRICS].mean().unstack("dataset")
    m = m.reindex([x for x in show if x in m.index])
    m.to_csv(out / "nocap_means.csv")
    # few blocks (2 datasets x 2 classifiers): use dataset x classifier x seed pairs, p-values indicative only
    tests = pd.concat([paired(g, "none", [x for x in show if x != "none"], met, unit=("dataset", "classifier", "seed"))
                       for met in METRICS], ignore_index=True)
    tests["mean_diff"] = -tests["mean_diff"]  # report method - none
    tests = tests.rename(columns={"wins": "none_wins", "losses": "none_losses", "mean_diff": "mean_diff_method_minus_none"})
    tests.to_csv(out / "nocap_tests.csv", index=False)
    imb = {}
    for ds in sorted(set(g.dataset)):
        p = Path(data_dir) / ds / "audit.json"
        if p.exists():
            cc = json.load(open(p)).get("class_counts", {})
            if cc:
                v = sorted(cc.values())
                imb[ds] = {"rows": int(sum(v)), "largest": int(v[-1]), "smallest": int(v[0]),
                           "imbalance_ratio": round(v[-1] / max(v[0], 1), 1)}
        r = g[g.dataset == ds].iloc[0]
        imb.setdefault(ds, {})["n_train_real"] = int(r.n_train_real)
    piv = g.groupby(["method", "dataset"]).macro_f1.mean().unstack().reindex([x for x in show if x in set(g.method)])
    to_tex(piv, out / "tab_nocap.tex",
           "Natural class imbalance: macro-F1 without the per-class cap and without the training-row cap "
           "(temporal protocol, XGBoost and MLP).", "tab:nocap")
    ax = g.drop_duplicates(["dataset", "seed", "method"]).groupby("method")[
        [c for c in ("validity_raw", "validity_final", "n_syn", "aug_wall_s") if c in g]].mean()
    ax.to_csv(out / "nocap_axes.csv")
    summary["nocap"] = {"seeds_completed_by_method": seeds_by_method, "seeds_used": sorted(int(x) for x in set(g.seed)),
                        "axes": ax.round(4).reset_index().to_dict("records"), "imbalance": imb, "tests": tests.round(4).to_dict("records"),
                        "macro_f1": piv.round(4).reset_index().to_dict("records")}
    LOG.info("no cap: %s\n%s\n%s", imb, piv.round(3).to_string(), tests.round(4).to_string(index=False))


# ------------------------------------------------------------------------------------------- tabsota
def block_tabsota(main, ts, out, summary):
    """Official TabDDPM / TabSyn (phase H) next to the phase-D methods, on the cells both phases share."""
    if not len(ts):
        return
    allr = pd.concat([main, ts], ignore_index=True).drop_duplicates(KEY, keep="last")
    g = allr[(allr.split == "temporal_stratified") & (allr.scenario == "full")]
    g = g[g.classifier.isin(set(ts.classifier)) & g.seed.isin(set(ts.seed))]
    show = ["none", "smote", "ctgan", "tvae", "tabddpm", "tabsyn", "diffusion", "validaug"]
    g = common_cells(g, [m for m in show if m in set(g.method)])
    if not len(g):
        LOG.warning("tabsota: no common cells with phase D"); return
    byds = g.groupby(["method", "dataset"]).macro_f1.mean().unstack()
    byds["mean"] = byds.mean(axis=1)
    byds = byds.reindex([m for m in show if m in byds.index])
    byds.to_csv(out / "tabsota_macro_f1_by_dataset.csv")
    t = mean_table(g, show).join(g.drop_duplicates(["dataset", "seed", "method"]).groupby("method")[
        [c for c in ("validity_raw", "c2st_auc", "wasserstein_mean", "aug_wall_s") if c in g]].mean())
    t.to_csv(out / "tabsota_means.csv")
    tests = pd.concat([paired(g, ref, [m for m in show if m not in ("tabddpm", "tabsyn")], met)
                       for ref in ("tabddpm", "tabsyn") if ref in set(g.method) for met in METRICS], ignore_index=True)
    tests.to_csv(out / "tabsota_tests.csv", index=False)
    to_tex(byds, out / "tab_tabsota.tex",
           "Official TabDDPM and TabSyn (macro-F1, temporal protocol, full data, cells shared with the main comparison).",
           "tab:tabsota")
    summary["tabsota"] = {"datasets": sorted(set(g.dataset)), "seeds": sorted(int(x) for x in set(g.seed)),
                          "means": t.round(4).reset_index().to_dict("records"), "tests": tests.round(4).to_dict("records")}
    LOG.info("TabDDPM / TabSyn:\n%s", byds.round(3).to_string())


# ------------------------------------------------------------------------------------------- c2st
def block_c2st(exp, out, summary, methods, seed, n_max, n_jobs):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    from validaug.encoder import TabularEncoder
    from validaug.runner import _load, _types, cap_train
    from validaug.splits import make_split

    cfg = yaml.safe_load(open(exp))
    root = Path(cfg["out_dir"])

    def auc_and_gain(X, y, cols):
        p = np.zeros(len(y)); gain = np.zeros(len(cols))
        for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(X, y):
            m = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.2, subsample=0.8, tree_method="hist",
                              n_jobs=n_jobs, random_state=0, verbosity=0, importance_type="total_gain")
            m.fit(X[a], y[a]); p[b] = m.predict_proba(X[b])[:, 1]; gain += m.feature_importances_
        return float(roc_auc_score(y, p)), gain / max(gain.sum(), 1e-12)

    rows, feats = [], []
    for ds in cfg["datasets"]:
        cell = f"{ds}__temporal_stratified__full__seed{seed}"
        sdir = root / "synthetic" / cell
        if not sdir.exists():
            LOG.warning("c2st: no saved synthetic samples in %s", sdir); continue
        df, rep = _load(cfg["data_dir"], ds)
        sp = root / "splits" / f"{ds}__temporal_stratified__{seed}.npz"
        tr = np.load(sp, allow_pickle=True)["train"] if sp.exists() else \
            make_split(df, "temporal_stratified", seed, train_conflicts=cfg.get("train_conflicts", "majority"))["train"]
        y_all = df["__y"].to_numpy(object)
        cap = cfg.get("max_train_rows")
        if cap and len(tr) > cap:
            tr = cap_train(tr, y_all, cap, seed)
        num, cat, bit = _types(df, rep)
        dtr = df.iloc[tr].reset_index(drop=True)
        enc = TabularEncoder(**cfg.get("encoder", {})).fit(dtr, num, cat, bit)
        # real flows go through the SAME encode/decode round trip as synthetic flows, so that quantisation
        # of the encoder (category bucketing, clipping, integer rounding) cannot separate the two samples
        real = enc.inverse_transform(enc.transform(enc.clean(dtr)))
        real["__y"] = y_all[tr]
        cols = enc.num_cols + enc.cat_cols + enc.bit_cols
        rng = np.random.default_rng(0)
        for m in methods:
            f = sdir / f"{m}.parquet"
            if not f.exists():
                continue
            syn = pd.read_parquet(f)
            parts_r, parts_s = [], []
            for c, gs in syn.groupby("__y"):  # same class mix on both sides
                gr = real[real["__y"] == c]
                n = min(len(gs), len(gr), max(50, n_max // max(1, syn["__y"].nunique())))
                if n < 20:
                    continue
                parts_s.append(gs.sample(n, random_state=0)); parts_r.append(gr.sample(n, random_state=0))
            if not parts_s:
                continue
            R, S = pd.concat(parts_r), pd.concat(parts_s)
            both = pd.concat([R[cols], S[cols]], ignore_index=True)
            for c in enc.cat_cols:
                both[c] = both[c].astype(str).astype("category").cat.codes
            X = both.astype("float32").fillna(-1).to_numpy()
            y = np.r_[np.zeros(len(R), int), np.ones(len(S), int)]
            auc, gain = auc_and_gain(X, y, cols)
            order = np.argsort(-gain)
            drop5 = np.setdiff1d(np.arange(len(cols)), order[:5]); drop10 = np.setdiff1d(np.arange(len(cols)), order[:10])
            auc5 = auc_and_gain(X[:, drop5], y, [cols[i] for i in drop5])[0] if len(drop5) > 1 else np.nan
            auc10 = auc_and_gain(X[:, drop10], y, [cols[i] for i in drop10])[0] if len(drop10) > 1 else np.nan
            num_ix = [i for i, c in enumerate(cols) if c in set(enc.num_cols)]
            # exact row-level novelty: share of synthetic numeric profiles that never occur among real flows
            rows.append({"dataset": ds, "method": m, "n_per_side": len(R), "c2st_auc": auc,
                         "c2st_auc_without_top5": auc5, "c2st_auc_without_top10": auc10,
                         "top_feature": cols[order[0]], "top_feature_gain_share": float(gain[order[0]]),
                         "top5_gain_share": float(gain[order[:5]].sum())})
            for rk, i in enumerate(order[:10]):
                c = cols[i]
                rv, sv = R[c].to_numpy(), S[c].to_numpy()
                novel = float((~pd.Series(sv).isin(pd.unique(rv))).mean())
                one = np.nan
                if i in num_ix:
                    x1 = both[[c]].astype("float32").fillna(-1).to_numpy()
                    one = auc_and_gain(x1, y, [c])[0]
                feats.append({"dataset": ds, "method": m, "rank": rk + 1, "feature": c, "gain_share": float(gain[i]),
                              "novel_value_rate": novel, "single_feature_auc": one,
                              "n_distinct_real": int(pd.Series(rv).nunique()), "n_distinct_syn": int(pd.Series(sv).nunique())})
            LOG.info("c2st [%s|%s] AUC=%.4f, without top-5 features %.4f, top feature %s (%.0f%% of gain)",
                     ds, m, auc, auc5, cols[order[0]], 100 * gain[order[0]])
    if rows:
        R = pd.DataFrame(rows); F = pd.DataFrame(feats)
        R.to_csv(out / "c2st_diagnosis.csv", index=False); F.to_csv(out / "c2st_top_features.csv", index=False)
        summary["c2st"] = {"per_cell": R.round(4).to_dict("records"),
                           "mean_by_method": R.groupby("method")[["c2st_auc", "c2st_auc_without_top5",
                                                                  "c2st_auc_without_top10", "top5_gain_share"]].mean().round(4)
                           .reset_index().to_dict("records"),
                           "most_frequent_top_features": F[F["rank"] <= 3].groupby(["method", "feature"]).size()
                           .sort_values(ascending=False).head(20).reset_index(name="n").to_dict("records")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default="runs/phaseD_main/results.jsonl")
    ap.add_argument("--cw", default="runs/phaseG_classweights/results.jsonl")
    ap.add_argument("--d50", default="runs/phaseG_diff50k/results.jsonl")
    ap.add_argument("--nocap", default="runs/phaseG_nocap/results.jsonl")
    ap.add_argument("--tabsota", default="runs/phaseH_tabsota/results.jsonl")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--c2st-exp", help="phase-D config; enables the real-vs-synthetic diagnosis")
    ap.add_argument("--c2st-methods", nargs="+", default=["smote", "diffusion", "diffusion_proj", "validaug"])
    ap.add_argument("--c2st-seed", type=int, default=0)
    ap.add_argument("--c2st-n", type=int, default=10000)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out", default="runs/phaseG_analysis")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", str(out / "analyze_controls.log"))
    warnings.filterwarnings("ignore")
    main_r = load(a.main)
    summary = {"n_rows": {"main": len(main_r)}}
    for name, path in (("cw", a.cw), ("d50", a.d50), ("nocap", a.nocap)):
        r = load(path)
        summary["n_rows"][name] = len(r)
        if not len(r):
            LOG.warning("%s: no results at %s -> skipped", name, path); continue
        if name == "cw":
            block_cw(main_r, r, out, summary)
        elif name == "d50":
            block_d50(main_r, r, out, summary)
        else:
            block_nocap(r, out, summary, a.data_dir)
    ts = load(a.tabsota)
    summary["n_rows"]["tabsota"] = len(ts)
    block_tabsota(main_r, ts, out, summary)
    if a.c2st_exp:
        block_c2st(a.c2st_exp, out, summary, a.c2st_methods, a.c2st_seed, a.c2st_n, a.n_jobs)
    dump_json(summary, out / "summary.json")
    LOG.info("written to %s", out)


if __name__ == "__main__":
    main()
