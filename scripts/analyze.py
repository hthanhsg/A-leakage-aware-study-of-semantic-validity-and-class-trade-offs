"""Phase analysis -> CSV/LaTeX tables, CD diagrams, statistical tests.

python scripts/analyze.py --results runs/full/results.jsonl --out runs/full/analysis
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validaug.stats import (cd_diagram, friedman_nemenyi, inflation, load_results, per_class_tradeoff,  # noqa: E402
                            to_latex_table, wilcoxon_holm)
from validaug.utils import dump_json  # noqa: E402

MAIN = ["none", "ros", "smote", "borderline_smote", "adasyn", "smote_enn", "ctgan", "tvae", "tabddpm", "tabsyn",
        "diffusion", "diffusion_proj", "validaug"]
ABL = ["validaug", "va_no_gc", "va_no_gb", "va_no_proj", "va_no_filter", "va_default_budget", "va_no_delta",
       "va_guidance_only", "diffusion"]
AXES = ["macro_f1", "rare_f1", "macro_f1_attacks", "macro_f1_common", "macro_pr_auc", "mcc", "binary_fpr", "validity_raw", "validity_final",
        "c2st_auc", "wasserstein_mean", "corr_diff", "dcr_ratio", "dcr_frac_exact_copy", "mia_auc", "aug_wall_s", "aug_energy_j", "n_syn"]


def order(cols, pref):
    pref = list(dict.fromkeys(pref))
    return [c for c in pref if c in cols] + [c for c in cols if c not in pref]


warnings.filterwarnings("ignore", category=RuntimeWarning)  # empty-slice means in sparse tables

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", default="validaug")
    ap.add_argument("--metrics", nargs="+", default=["macro_f1", "rare_f1", "macro_f1_common", "mcc", "macro_pr_auc"])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = load_results(a.results)
    print(f"{len(df)} result rows; datasets={df.dataset.nunique()} methods={df.method.nunique()}")
    stats_all = {}
    for metric in a.metrics:
        if metric not in df:
            continue
        # ---- main tables: one per (split, scenario), averaged over classifiers & seeds
        for (split, scen), g in df.groupby(["split", "scenario"]):
            per_seed = g.groupby(["dataset", "method", "seed"])[metric].mean().reset_index()
            mean = per_seed.pivot_table(index="dataset", columns="method", values=metric, aggfunc="mean")
            if mean.empty:
                continue
            std = per_seed.pivot_table(index="dataset", columns="method", values=metric, aggfunc="std")
            cols = order(list(mean.columns), MAIN + ABL)
            mean, std = mean[cols], std.reindex(columns=cols)
            mean.loc["Average"] = mean.mean()
            tag = f"{metric}__{split}__{scen}"
            mean.round(4).to_csv(out / f"table_{tag}.csv")
            main_cols = [c for c in cols if c in MAIN]
            if main_cols:
                to_latex_table(mean[main_cols], std.reindex(index=mean.index, columns=main_cols), str(out / f"table_{tag}.tex"),
                               f"{metric.replace('_', ' ')} ({split}, {scen}); mean $\\pm$ std over seeds.",
                               f"tab:{tag}")
            abl_cols = [c for c in cols if c in ABL]
            if len(abl_cols) > 2:
                to_latex_table(mean[abl_cols], None, str(out / f"ablation_{tag}.tex"),
                               f"Ablation VALID-Aug — {metric.replace('_', ' ')} ({split}, {scen}).", f"tab:abl:{tag}")
        # ---- block-level statistics, per protocol
        for split, g in df.groupby("split"):
            meths = [m for m in MAIN if m in g.method.unique()]
            res = friedman_nemenyi(g, metric, meths)
            if res:
                stats_all[f"{metric}__{split}"] = res
                cd_diagram(res, str(out / f"cd_{metric}__{split}.png"),
                           f"{metric} — {split} (N={res['n_blocks']}, Friedman p={res['friedman_p']:.2g})")
            w = wilcoxon_holm(g, metric, a.control, meths)
            if len(w):
                w.to_csv(out / f"wilcoxon_{metric}__{split}.csv", index=False)
        inf = inflation(df, metric)
        if len(inf):
            inf.round(4).to_csv(out / f"inflation_{metric}.csv", index=False)
    dump_json(stats_all, out / "friedman_nemenyi.json")

    # ---- per-class trade-off vs `none` (gains on rare classes vs losses on other classes)
    tr = per_class_tradeoff(df)
    if len(tr):
        tr.to_csv(out / "tradeoff_per_cell.csv", index=False)
        agg = tr.groupby(["split", "scenario", "method"])[["n_classes", "n_gain", "n_loss", "mean_gain",
                                                          "mean_loss", "worst_drop"]].mean().round(4)
        agg.to_csv(out / "tradeoff_summary.csv")
        tr.groupby(["dataset", "split", "method"])["worst_class"].agg(lambda x: x.value_counts().index[0]) \
          .to_csv(out / "tradeoff_most_hurt_class.csv")

    # ---- four-axis table (averaged over all blocks) per method
    axes = [c for c in AXES if c in df.columns]
    t = df.groupby("method")[axes].mean()
    t = t.loc[order(list(t.index), MAIN + ABL)]
    t.round(4).to_csv(out / "four_axes_by_method.csv")
    # per protocol
    df.groupby(["split", "method"])[axes].mean().round(4).to_csv(out / "four_axes_by_split_method.csv")
    # holdout / cross-dataset
    for col in ["holdout_detection", "binary_tpr", "binary_fpr", "tpr_at_fpr1", "binary_roc_auc"]:
        if col in df:
            sub = df[df[col].notna()]
            if len(sub):
                sub.pivot_table(index=["dataset", "split", "scenario"], columns="method", values=col).round(4) \
                   .to_csv(out / f"table_{col}.csv")
    print("Analysis written to", out)
