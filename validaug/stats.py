"""Statistical tests & result tables (following Demsar, JMLR 2006; Benavoli et al., 2016).

* Friedman + Nemenyi (critical-difference diagram) over blocks (dataset x split x scenario x classifier).
* Paired Wilcoxon signed-rank tests (VALID-Aug vs each method) + Holm correction + rank-biserial r.
* "Inflation" index: DA gain under the random split minus DA gain under leakage-free splits.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

Q_NEMENYI_05 = [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164, 3.219, 3.268, 3.313,
                3.354, 3.391, 3.426, 3.458, 3.489, 3.517, 3.544]  # k = 2..20
BLOCK = ["dataset", "split", "scenario", "classifier"]


def load_results(path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").drop_duplicates(["dataset", "split", "scenario", "seed", "method", "classifier"], keep="last")
    # back-fill metrics added in later code versions (computable from stored per-class F1)
    def _att(r):
        f = r.get("f1_per_class")
        return float(np.mean([v for c, v in f.items() if c != "Benign"])) if isinstance(f, dict) and len(f) > 1 else np.nan

    def _com(r):
        f, rare = r.get("f1_per_class"), r.get("rare_classes") or []
        if not isinstance(f, dict):
            return np.nan
        v = [x for c, x in f.items() if c != "Benign" and c not in rare]
        return float(np.mean(v)) if v else np.nan
    for col, fn in (("macro_f1_attacks", _att), ("macro_f1_common", _com)):
        if col not in df:
            df[col] = np.nan
        miss = df[col].isna()
        if miss.any():
            df.loc[miss, col] = df[miss].apply(fn, axis=1)
    return df


def seed_mean(df, metric, by=("dataset", "split", "scenario", "classifier", "method")):
    g = df.groupby(list(by))[metric]
    return g.mean().rename("mean").to_frame().join(g.std().rename("std")).join(g.count().rename("n")).reset_index()


def block_matrix(df, metric, methods=None, block=BLOCK):
    m = df.groupby(block + ["method"])[metric].mean().unstack("method")
    if methods:
        m = m[[c for c in methods if c in m.columns]]
    return m.dropna()


def friedman_nemenyi(df, metric, methods=None, block=BLOCK, higher_better=True):
    M = block_matrix(df, metric, methods, block)
    if M.shape[1] < 2 or M.shape[0] < 2:
        return None
    ranks = (-M if higher_better else M).rank(axis=1, method="average")
    avg = ranks.mean().sort_values()
    k, n = M.shape[1], M.shape[0]
    stat, p = friedmanchisquare(*[M[c].to_numpy() for c in M.columns]) if k >= 3 else (np.nan, np.nan)
    cd = Q_NEMENYI_05[k - 2] * np.sqrt(k * (k + 1) / (6.0 * n)) if k <= 20 else np.nan
    return {"metric": metric, "n_blocks": n, "k": k, "friedman_chi2": float(stat), "friedman_p": float(p),
            "cd": float(cd), "avg_rank": avg.to_dict()}


def wilcoxon_holm(df, metric, control="validaug", methods=None, block=BLOCK):
    M = block_matrix(df, metric, methods, block)
    if control not in M.columns:
        return pd.DataFrame()
    out = []
    for c in M.columns:
        if c == control:
            continue
        d = (M[control] - M[c]).to_numpy()
        nz = d[d != 0]
        if len(nz) < 2:
            out.append({"vs": c, "n": len(d), "median_diff": float(np.median(d)), "p": 1.0, "rank_biserial": 0.0}); continue
        _, p = wilcoxon(nz)
        r = pd.Series(np.abs(nz)).rank().to_numpy()
        wp, wm = r[nz > 0].sum(), r[nz < 0].sum()
        out.append({"vs": c, "n": len(d), "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                    "losses": int((d < 0).sum()), "median_diff": float(np.median(d)), "mean_diff": float(d.mean()),
                    "p": float(p), "rank_biserial": float((wp - wm) / (wp + wm))})
    res = pd.DataFrame(out).sort_values("p").reset_index(drop=True)
    m = len(res)
    adj, running = [], 0.0
    for i, p in enumerate(res["p"]):
        running = max(running, min(1.0, (m - i) * p))
        adj.append(running)
    res["p_holm"] = adj
    res["significant_05"] = res["p_holm"] < 0.05
    return res


def inflation(df, metric, leak_free=("random", "temporal_stratified", "group"), refs=("random_naive", "random"),
              baseline="none"):
    """Inflation of a reference protocol over a stricter one.

    * absolute: score(none | ref) - score(none | strict)
    * DA gain : [metric(method) - metric(none)] under ref minus the same gain under strict,
      paired within (dataset, scenario, classifier, seed).
    Pairs: random_naive -> random (duplicate leakage), random -> temporal/group (temporal/host leakage),
    random_naive -> temporal/group (total).
    """
    keys = ["dataset", "scenario", "classifier", "seed"]
    piv = df.pivot_table(index=keys + ["split"], columns="method", values=metric)
    if baseline not in piv.columns:
        return pd.DataFrame()
    gain = piv.sub(piv[baseline], axis=0).drop(columns=[baseline]).reset_index()
    pb = piv[baseline].unstack("split")
    rows = []
    for ref in refs:
        for lf in leak_free:
            if lf == ref or ref not in pb or lf not in pb:
                continue
            both = pb[[ref, lf]].dropna()
            if len(both):
                rows.append({"reference": ref, "strict": lf, "method": f"[{baseline}] absolute score",
                             "value_reference": both[ref].mean(), "value_strict": both[lf].mean(),
                             "inflation": (both[ref] - both[lf]).mean(), "n": len(both)})
            a = gain[gain.split == ref].set_index(keys).drop(columns="split")
            b = gain[gain.split == lf].set_index(keys).drop(columns="split")
            common = a.index.intersection(b.index)
            for meth in a.columns:
                ga, gb = a.loc[common, meth].dropna(), b.loc[common, meth].dropna()
                ix = ga.index.intersection(gb.index)
                if len(ix):
                    rows.append({"reference": ref, "strict": lf, "method": meth,
                                 "value_reference": ga[ix].mean(), "value_strict": gb[ix].mean(),
                                 "inflation": (ga[ix] - gb[ix]).mean(), "n": len(ix)})
    return pd.DataFrame(rows)


def per_class_tradeoff(df, baseline="none", tol=0.01):
    """Per-class F1 change of every method vs `baseline`, paired within
    (dataset, split, scenario, classifier, seed): how many classes gain / lose more than `tol`,
    the mean gain on gaining classes, the mean loss on losing classes, and the worst single drop."""
    keys = ["dataset", "split", "scenario", "classifier", "seed"]
    base = df[df.method == baseline].set_index(keys)["f1_per_class"]
    rows = []
    for _, r in df[df.method != baseline].iterrows():
        k = tuple(r[c] for c in keys)
        if k not in base.index or not isinstance(r["f1_per_class"], dict):
            continue
        b = base.loc[k]
        b = b.iloc[0] if isinstance(b, pd.Series) else b
        d = {c: r["f1_per_class"][c] - b[c] for c in r["f1_per_class"] if c in b}
        v = np.array(list(d.values()))
        rows.append({**dict(zip(keys, k)), "method": r["method"], "n_classes": len(v),
                     "n_gain": int((v > tol).sum()), "n_loss": int((v < -tol).sum()),
                     "mean_gain": float(v[v > tol].mean()) if (v > tol).any() else 0.0,
                     "mean_loss": float(v[v < -tol].mean()) if (v < -tol).any() else 0.0,
                     "worst_drop": float(v.min()) if len(v) else np.nan,
                     "worst_class": min(d, key=d.get) if d else None})
    return pd.DataFrame(rows)


def cd_diagram(res: dict, path: str, title: str = ""):
    """Demsar-style (2006) critical-difference diagram: rank 1 (best) on the left."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avg = pd.Series(res["avg_rank"]).sort_values()
    names, vals = list(avg.index), avg.to_numpy()
    k, cd = len(vals), res["cd"]
    half = (k + 1) // 2
    # maximal cliques with no significant difference
    cliques, last = [], -1
    for i in range(k):
        j = int(np.searchsorted(vals, vals[i] + cd + 1e-12, side="right") - 1) if np.isfinite(cd) else i
        if j > i and j > last:
            cliques.append((i, j)); last = j
    h = 1.2 + 0.32 * max(half, 1) + 0.18 * len(cliques)
    fig, ax = plt.subplots(figsize=(8, h))
    lo, hi = 1, max(k, 2)
    ax.set_xlim(lo - 1.6, hi + 1.6); ax.set_ylim(-(half + 1) * 0.32 - 0.18 * len(cliques) - 0.2, 0.75)
    ax.plot([lo, hi], [0, 0], color="k", lw=1)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [0, 0.08], color="k", lw=1)
        ax.text(r, 0.14, str(r), ha="center", va="bottom", fontsize=8)
    if np.isfinite(cd):
        ax.plot([lo, lo + cd], [0.5, 0.5], color="C3", lw=2)
        ax.text(lo + cd / 2, 0.56, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8, color="C3")
    base = -0.18 * len(cliques) - 0.25
    for i, (n, v) in enumerate(zip(names, vals)):
        if i < half:
            y = base - 0.32 * i; x_end = lo - 0.15; ha = "right"
        else:
            y = base - 0.32 * (k - 1 - i); x_end = hi + 0.15; ha = "left"
        ax.plot([v, v, x_end], [0, y, y], color="0.35", lw=0.8)
        ax.text(x_end + (-0.05 if ha == "right" else 0.05), y, f"{n} ({v:.2f})", ha=ha, va="center", fontsize=8)
    for lvl, (i, j) in enumerate(cliques):
        y = -0.12 - 0.18 * lvl
        ax.plot([vals[i] - 0.03, vals[j] + 0.03], [y, y], color="k", lw=3, solid_capstyle="round")
    ax.axis("off"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def to_latex_table(mean: pd.DataFrame, std: pd.DataFrame | None, path: str, caption: str, label: str,
                   higher_better=True, digits=4):
    """Booktabs table: rows = dataset, columns = method; best value per row in bold."""
    cols = list(mean.columns)
    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Dataset & " + " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\", "\\midrule"]
    for idx, row in mean.iterrows():
        best = row.max() if higher_better else row.min()
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("--"); continue
            s = f"{v:.{digits}f}"
            if std is not None and c in std.columns and pd.notna(std.loc[idx, c]):
                s += f"$\\pm${std.loc[idx, c]:.{digits}f}"
            cells.append(f"\\textbf{{{s}}}" if np.isclose(v, best) else s)
        name = idx if isinstance(idx, str) else " / ".join(map(str, idx))
        lines.append(name.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
