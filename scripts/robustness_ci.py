"""Bootstrap intervals that account for the dependence between blocks of the same dataset.

Computed from the stored results only (no training):
  1. protocol effect (phase B): drop of macro-F1 of the unaugmented classifiers from the
     deduplicated random protocol to the temporal and group protocols, 90 paired cells;
  2. main comparison (phase D, temporal, full data): mean difference to no augmentation over
     the 30 dataset--classifier blocks, for macro-F1 and MCC;
  3. seed variability of macro-F1 within blocks.
Intervals are percentile intervals over (a) cells/blocks and (b) whole datasets (cluster bootstrap).

    python scripts/robustness_ci.py [--runs runs] [--B 20000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])


def boot_cells(x, rng, B):
    v = np.asarray(x)
    return np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)], [2.5, 97.5])


def boot_datasets(x, rng, B):
    """x: Series indexed by (dataset, ...); resample datasets with replacement."""
    ds = list(x.index.get_level_values(0).unique())
    arr = {k: x.xs(k, level=0).values for k in ds}
    means = [np.concatenate([arr[k] for k in rng.choice(ds, len(ds))]).mean() for _ in range(B)]
    return np.percentile(means, [2.5, 97.5])


def fmt(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    runs = Path(a.runs)

    print("== Protocol effect (phase B, method none, macro-F1, random minus stricter protocol)")
    b = load(runs / "phaseB_protocol" / "results.jsonl")
    p = b[b.method == "none"].pivot_table(index=["dataset", "classifier", "seed", "scenario"],
                                          columns="split", values="macro_f1")
    for s in ["temporal_stratified", "group"]:
        rng = np.random.default_rng(a.seed)
        d = (p["random"] - p[s]).dropna()
        print(f"{s:20s} n={len(d)} mean={d.mean():.3f} cells {fmt(boot_cells(d, rng, a.B))} "
              f"datasets {fmt(boot_datasets(d, rng, a.B))}")

    print("== Main comparison (phase D, temporal, full data; method minus none over 30 blocks)")
    d = load(runs / "phaseD_main" / "results.jsonl")
    d = d[(d.split == "temporal_stratified") & (d.scenario == "full") & ~d.method.isin(["tabddpm", "tabsyn"])]
    for met in ["macro_f1", "mcc"]:
        blk = d.pivot_table(index=["dataset", "classifier"], columns="method", values=met, aggfunc="mean")
        for m in ["smote", "ros", "diffusion", "validaug"]:
            rng = np.random.default_rng(a.seed)
            x = blk[m] - blk["none"]
            print(f"{met:8s} {m:10s} mean={x.mean():+.4f} blocks {fmt(boot_cells(x, rng, a.B))} "
                  f"datasets {fmt(boot_datasets(x, rng, a.B))}")

    print("== Seed standard deviation of macro-F1 within dataset--classifier blocks")
    sd = d.groupby(["method", "dataset", "classifier"]).macro_f1.std()
    for m in ["none", "smote", "validaug"]:
        print(f"{m:10s} mean SD={sd[m].mean():.4f} max SD={sd[m].max():.4f}")
    print(f"all methods: mean SD={sd.mean():.4f}, max SD={sd.max():.4f}")


if __name__ == "__main__":
    main()
