#!/usr/bin/env python3
"""Redraw every data figure of the paper from the raw per-run results and the mechanism CSV files.

    python scripts/make_figures.py --runs runs --mechanism runs/mechanism --out figures

Input
    runs/phaseB_protocol/results.jsonl, runs/phaseD_main/results.jsonl      (Figures 2-6)
    runs/mechanism/{ceiling,shift,overlap_vs_delta,overlap_correlation}.csv  (Figures 7, 8, S1)
Output (figures/)
    Figure_2.pdf   macro-F1 without augmentation under the four splitting protocols
    Figure_3.pdf   change brought by five oversamplers under each protocol (macro-F1, rare-F1, MCC)
    Figure_4.pdf   per-class trade-off: classes improved / degraded and worst per-class drop
    Figure_5.pdf   scarce regimes: change of macro-F1 and rare-F1 relative to no augmentation
    Figure_6.pdf   VALID-Aug: gain predicted on validation vs. gain observed on test (84 cells)
    Figure_7.pdf   label-conflict upper bound of macro-F1 vs. best unaugmented classifier
    Figure_8.pdf   macro-F1 without augmentation vs. class-conditional train-test shift
    Figure_S1.pdf  per-class F1 change vs. class overlap (supplementary material)
Figure 1 (overview of the probe) is a TikZ drawing: see Figure_1.tex.
The numbers behind Figures 2-6 are printed in Tables S10-S14 of the supplementary material and
are checked against the same files by scripts/verify_tables.py.

Only pandas, numpy, scipy and matplotlib are required; no model is trained and no dataset is read.
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

DS_SHORT = {"nf_unsw_v3": "UNSW", "nf_ton_v3": "ToN", "nf_bot_v3": "BoT", "nf_cic2018_v3": "CIC18", "cic2017_fixed": "CIC17"}
SPLIT_SHORT = {"random_naive": "Naive", "random": "Random", "temporal_stratified": "Temporal", "group": "Group"}
METHOD_NAME = {"smote": "SMOTE", "ctgan": "CTGAN", "diffusion": "Diffusion", "validaug": "VALID-Aug"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
HUE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]     # fixed colour per dataset (order of DS_SHORT)
MARK = ["o", "s", "^", "D", "v"]                                  # and a marker, so colour is never the only cue
STYLE = {d: (HUE[i], MARK[i]) for i, d in enumerate(DS_SHORT)}

plt.rcParams.update({"font.size": 8, "hatch.linewidth": 0.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})


METHODS = ["ros", "smote", "borderline_smote", "adasyn", "smote_enn", "ctgan", "tvae", "diffusion", "diffusion_proj", "validaug"]
MLABEL = {"ros": "ROS", "smote": "SMOTE", "borderline_smote": "Borderline-SMOTE", "adasyn": "ADASYN", "smote_enn": "SMOTE-ENN",
          "ctgan": "CTGAN", "tvae": "TVAE", "diffusion": "Diffusion", "diffusion_proj": "Diffusion+Proj", "validaug": "VALID-Aug"}
SPLITS = ["random_naive", "random", "temporal_stratified", "group"]
SPLIT_LONG = {"random_naive": "Naive random", "random": "Random", "temporal_stratified": "Temporal", "group": "Group"}


def load(runs, phase):
    return pd.DataFrame([json.loads(l) for l in open(Path(runs) / phase / "results.jsonl") if l.strip()])


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1, run)
    return adj


def figure_protocol(B, path):
    """Macro-F1 of the unaugmented classifiers under the four protocols (multi-class; mean over three
    classifiers and three seeds). One line per dataset."""
    d = B[(B.method == "none") & (B.scenario == "full")]
    g = d.groupby(["dataset", "split"]).macro_f1.mean().unstack()[SPLITS]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    x = np.arange(len(SPLITS))
    for ds in DS_SHORT:
        col, mk = STYLE[ds]
        ax.plot(x, g.loc[ds], color=col, lw=1.2, alpha=0.6, zorder=1)
        ax.scatter(x, g.loc[ds], color=col, marker=mk, s=26, edgecolor="white", linewidth=0.6, label=DS_SHORT[ds], zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([SPLIT_LONG[s] for s in SPLITS]); ax.set_xlim(-0.3, len(SPLITS) - 0.7)
    ax.set_ylabel("Macro-F1 without augmentation"); ax.set_ylim(0.45, 1.02); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.02), columnspacing=1.0, handletextpad=0.3, borderaxespad=0.0)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def figure_oversamplers(B, path):
    """Change relative to no augmentation for five oversamplers (multi-class, full data; mean over the
    15 dataset-classifier blocks). Filled marker: Holm-adjusted Wilcoxon p < 0.05 within a protocol and metric."""
    ms = METHODS[:5]
    metrics = [("macro_f1", "Macro-F1"), ("rare_f1", "Rare-F1"), ("mcc", "MCC")]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6), sharey=True)
    y = np.arange(len(ms))[::-1]
    off = np.linspace(0.27, -0.27, len(SPLITS))
    for ax, (col, title) in zip(axes, metrics):
        for k, sp in enumerate(SPLITS):
            d = B[(B.scenario == "full") & (B.split == sp)]
            b = d.groupby(["method", "dataset", "classifier"])[col].mean().unstack("method")
            delta = [(b[m] - b["none"]).mean() for m in ms]
            sig = holm([wilcoxon(b[m], b["none"]).pvalue for m in ms]) < 0.05
            for yy, dv, sg in zip(y, delta, sig):
                ax.scatter(dv, yy + off[k], marker=MARK[k], s=22, linewidth=0.9, zorder=2,
                           facecolor=HUE[k] if sg else "white", edgecolor=HUE[k])
            ax.scatter([], [], marker=MARK[k], s=22, facecolor=HUE[k], edgecolor=HUE[k], label=SPLIT_LONG[sp])
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        ax.set_title(title, fontsize=8, color=INK); ax.set_xlabel("Change vs. no augmentation")
        ax.set_yticks(y); ax.set_yticklabels([MLABEL[m] for m in ms]); ax.grid(axis="y", visible=False)
        ax.set_xlim(-0.042, 0.022); ax.set_xticks([-0.04, -0.02, 0.0, 0.02])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.03), columnspacing=1.2, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _temporal_full(D):
    return D[(D.split == "temporal_stratified") & (D.scenario == "full") & D.method.isin(METHODS + ["none"])]


def figure_tradeoff(D, path):
    """Per-class trade-off relative to no augmentation (temporal protocol, full data): mean number of classes
    whose F1 changes by more than 0.01, and mean of the largest per-class drop."""
    tf = _temporal_full(D)
    base = tf[tf.method == "none"].set_index(["dataset", "classifier", "seed"])
    rec = []
    for _, r in tf[tf.method != "none"].iterrows():
        k = (r.dataset, r.classifier, r.seed)
        if k not in base.index:
            continue
        f0 = base.loc[k, "f1_per_class"]
        dl = [r.f1_per_class[c] - f0[c] for c in r.f1_per_class if c in f0]
        rec.append((r.method, sum(v > 0.01 for v in dl), sum(v < -0.01 for v in dl), min(dl)))
    t = pd.DataFrame(rec, columns=["m", "imp", "deg", "worst"]).groupby("m").mean().loc[METHODS]
    y = np.arange(len(METHODS))[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
    # grey-scale safe: improved = solid dark fill, degraded = white fill with hatching; the two
    # also sit on opposite sides of zero, so neither colour nor position is the only cue
    a.barh(y, t["imp"], height=0.62, color="#2a78d6", edgecolor=INK, linewidth=0.6, label="Classes improved")
    a.barh(y, -t["deg"], height=0.62, color="white", edgecolor="#c4501f", hatch="//////", linewidth=0.6,
           label="Classes degraded")
    for yy, i, dg in zip(y, t["imp"], t["deg"]):
        a.text(i + 0.06, yy, "%.2f" % i, va="center", ha="left", fontsize=6.5, color=INK)
        a.text(-dg - 0.06, yy, "%.2f" % dg, va="center", ha="right", fontsize=6.5, color=INK)
    a.axvline(0, color=MUTED, lw=0.8); a.set_xlim(-4.1, 3.1)
    a.set_xticks([-3, -2, -1, 0, 1, 2, 3]); a.set_xticklabels(["3", "2", "1", "0", "1", "2", "3"])
    a.set_xlabel("Mean number of classes with |$\\Delta$F1| > 0.01")
    a.set_yticks(y); a.set_yticklabels([MLABEL[m] for m in METHODS]); a.grid(axis="y", visible=False)
    a.legend(fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), columnspacing=1.2, handletextpad=0.4)
    b.barh(y, t["worst"], height=0.62, color="white", edgecolor="#c4501f", hatch="//////", linewidth=0.6)
    for yy, w in zip(y, t["worst"]):
        b.text(w - 0.004, yy, "%.3f" % w, va="center", ha="right", fontsize=6.5, color=INK)
    b.set_xlim(-0.245, 0); b.set_xlabel("Mean worst per-class change of F1"); b.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def figure_scarce(D, path):
    """Scarce regimes (temporal protocol; five datasets, three classifiers, three seeds): change of macro-F1
    and rare-F1 relative to no augmentation. CTGAN and TVAE were run for the 5% scenario only."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    y = np.arange(len(METHODS))[::-1]
    for ax, (sc, title) in zip(axes, [("rare5", "Rare classes at 5%"), ("rare1", "Rare classes at 1%")]):
        d = D[(D.split == "temporal_stratified") & (D.scenario == sc)]
        n = d[d.method == "none"]
        for yy, m in zip(y, METHODS):
            r = d[d.method == m]
            if len(r) < len(n):                       # incomplete (CTGAN/TVAE at 1%): not drawn
                ax.text(0.0, yy, "not run", va="center", ha="center", fontsize=6.5, color=MUTED,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
                continue
            dm, dr = r.macro_f1.mean() - n.macro_f1.mean(), r.rare_f1.mean() - n.rare_f1.mean()
            ax.plot([dm, dr], [yy, yy], color=GRID, lw=1.2, zorder=1)
            ax.scatter(dm, yy, marker="o", s=24, color="#2a78d6", edgecolor="white", linewidth=0.6, zorder=3)
            ax.scatter(dr, yy, marker="D", s=20, facecolor="white", edgecolor="#eb6834", linewidth=1.0, zorder=2)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        ax.set_title(title, fontsize=8, color=INK); ax.set_xlabel("Change vs. no augmentation")
        ax.set_yticks(y); ax.set_yticklabels([MLABEL[m] for m in METHODS]); ax.grid(axis="y", visible=False)
        ax.set_xlim(-0.055, 0.075)
    axes[0].scatter([], [], marker="o", s=24, color="#2a78d6", label="Macro-F1")
    axes[0].scatter([], [], marker="D", s=20, facecolor="white", edgecolor="#eb6834", linewidth=1.0, label="Rare-F1")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.03), columnspacing=1.2, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def budget_cells(D):
    """One row per dataset-protocol-scenario-seed cell of the main comparison: macro-F1 gain predicted on
    validation by the budget search, and gain observed on test (mean over the classifiers of the cell)."""
    rows = []
    for (ds, sp, sc, seed), g in D[D.method == "validaug"].groupby(["dataset", "split", "scenario", "seed"]):
        ai = g.iloc[0].aug_info
        bud = ai.get("budget") if isinstance(ai, dict) else None
        if not isinstance(bud, dict) or "gain" not in bud:
            continue
        n = D[(D.method == "none") & (D.dataset == ds) & (D.split == sp) & (D.scenario == sc) & (D.seed == seed)]
        tg = g.set_index("classifier").macro_f1.sub(n.set_index("classifier").macro_f1).dropna().mean()
        rows.append((ds, sp, sc, seed, bud["gain"], tg))
    return pd.DataFrame(rows, columns=["dataset", "split", "scenario", "seed", "val_gain", "test_gain"])


def figure_budget(D, path):
    """Validation-predicted against test-observed gain of VALID-Aug; points below the diagonal are cells in
    which the validation gain overstates the test gain."""
    C = budget_cells(D)
    # grey-scale safe: every group has its own marker shape; fill (solid/open) separates full from scarce data
    groups = [("temporal_stratified", "full", "Temporal, full", HUE[0], "o", True),
              ("temporal_stratified", "rare5", "Temporal, 5%", HUE[0], "s", False),
              ("temporal_stratified", "rare1", "Temporal, 1%", HUE[0], "^", False),
              ("temporal_stratified", "binary", "Temporal, binary", HUE[0], "D", False),
              ("group", "full", "Group, full", HUE[1], "v", True),
              ("random", "full", "Random, full", HUE[2], "X", True)]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    lo, hi = -0.07, 0.13
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    for sp, sc, lab, col, mk, filled in groups:
        g = C[(C.split == sp) & (C.scenario == sc)]
        ax.scatter(g.val_gain, g.test_gain, marker=mk, s=24 if mk == "X" else 20, facecolor=col if filled else "white",
                   edgecolor=INK if filled else col, linewidth=0.5 if filled else 0.9, alpha=0.9, label=lab, zorder=2)
    ax.set_xlim(-0.005, 0.105); ax.set_ylim(lo, hi)
    ax.set_xlabel("Macro-F1 gain predicted on validation"); ax.set_ylabel("Macro-F1 gain observed on test")
    ax.text(0.094, 0.103, "y = x", fontsize=6.5, color=MUTED, rotation=38, ha="center", va="center")
    ax.legend(fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), columnspacing=0.9, handletextpad=0.2, borderaxespad=0.0)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)



def figure_ceiling(ceil, path):
    """Bars: upper bound of macro-F1 set by label-conflicting test flows (light) and the best
    unaugmented classifier (dark), one panel per splitting protocol."""
    c = ceil[ceil["none_macro_f1_best"].notna()]
    splits = [s for s in SPLIT_SHORT if s in set(c["split"])]
    fig, axes = plt.subplots(1, len(splits), figsize=(1.75 * len(splits), 2.4), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], splits):
        g = c[c["split"] == s].set_index("dataset")
        g = g.loc[[d for d in DS_SHORT if d in g.index]]
        x = np.arange(len(g))
        # grey-scale safe: two bars side by side instead of one drawn over the other, so a bound
        # equal to the classifier score stays visible; hatched outline vs. solid fill
        w = 0.38
        ax.bar(x - w / 2, g["macro_f1_upper_bound"], width=w, color="white", edgecolor=INK, hatch="//////",
               linewidth=0.6, label="Upper bound (label conflicts)")
        ax.bar(x + w / 2, g["none_macro_f1_best"], width=w, color="#2a78d6", edgecolor=INK, linewidth=0.6,
               label="Best classifier, no augmentation")
        ax.set_xticks(x); ax.set_xticklabels([DS_SHORT[d] for d in g.index], rotation=45, ha="right")
        ax.set_title(SPLIT_SHORT[s], fontsize=8, color=INK); ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Macro-F1"); axes[0][0].set_ylim(0, 1.02)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def figure_shift(shift, path):
    """One point per dataset and protocol; lines connect the protocols of a dataset."""
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for d in DS_SHORT:
        g = shift[shift["dataset"] == d].sort_values("c2st_auc_classcond")
        if not len(g):
            continue
        col, mk = STYLE[d]
        ax.plot(g["c2st_auc_classcond"], g["none_macro_f1"], color=col, lw=1, alpha=0.5, zorder=1)
        ax.scatter(g["c2st_auc_classcond"], g["none_macro_f1"], color=col, marker=mk, s=22,
                   edgecolor="white", linewidth=0.6, label=DS_SHORT[d], zorder=2)
    ax.set_xlabel("Class-conditional train-test C2ST AUC")
    ax.set_ylabel("Macro-F1 without augmentation")
    # legend above the axes: inside the plot it would cover data points
    ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.02), columnspacing=1.0,
              handletextpad=0.3, borderaxespad=0.0)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def figure_s1(octab, corr, path):
    """Per-class F1 change against class overlap for four methods; one point per dataset-class pair."""
    show = [m for m in METHOD_NAME if m in set(octab["method"])]
    fig, axes = plt.subplots(1, len(show), figsize=(1.8 * len(show), 2.5), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], show):
        g = octab[octab["method"] == m]
        for d in DS_SHORT:
            gg = g[g["dataset"] == d]
            if len(gg):
                ax.scatter(gg["overlap"], gg["delta_f1"], color=STYLE[d][0], marker=STYLE[d][1], s=16,
                           edgecolor="white", linewidth=0.5, label=DS_SHORT[d])
        ax.axhline(0, color=MUTED, lw=0.8)
        rr = corr[(corr["method"] == m) & (corr["x"] == "overlap") & (corr["variant"] == "pooled")]
        title = METHOD_NAME[m] + ("" if not len(rr) else " ($\\rho$ = %.2f)" % rr["rho"].iloc[0])
        ax.set_title(title, fontsize=8, color=INK); ax.set_xlabel("Class overlap")
    axes[0][0].set_ylabel("$\\Delta$F1 of the class vs. no augmentation")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", help="directory with phaseB_protocol/ and phaseD_main/ results.jsonl")
    ap.add_argument("--mechanism", default="runs/mechanism", help="directory with the CSV files of mechanism_analysis.py")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    src, out = Path(a.mechanism), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B, D = load(a.runs, "phaseB_protocol"), load(a.runs, "phaseD_main")
    figure_protocol(B, out / "Figure_2.pdf")
    figure_oversamplers(B, out / "Figure_3.pdf")
    figure_tradeoff(D, out / "Figure_4.pdf")
    figure_scarce(D, out / "Figure_5.pdf")
    figure_budget(D, out / "Figure_6.pdf")
    budget_cells(D).to_csv(out / "Figure_6_data.csv", index=False)
    figure_ceiling(pd.read_csv(src / "ceiling.csv"), out / "Figure_7.pdf")
    figure_shift(pd.read_csv(src / "shift.csv"), out / "Figure_8.pdf")
    figure_s1(pd.read_csv(src / "overlap_vs_delta.csv"), pd.read_csv(src / "overlap_correlation.csv"), out / "Figure_S1.pdf")
    for f in sorted(out.glob("Figure_*.pdf")):
        print("written", f)


if __name__ == "__main__":
    main()
