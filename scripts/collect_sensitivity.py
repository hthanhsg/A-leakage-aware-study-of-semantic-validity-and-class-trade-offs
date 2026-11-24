"""Collect sensitivity-analysis results (runs/phaseE_sens/<param>_<value>/results.jsonl) + default values
(from runs/phaseE_ablation) -> CSV table and line plots per hyper-parameter.

python scripts/collect_sensitivity.py --root runs/phaseE_sens --default runs/phaseE_ablation/results.jsonl
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validaug.stats import load_results  # noqa: E402

DEFAULTS = {"lambda_c": 1.0, "lambda_b": 0.2, "beta": 10}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/phaseE_sens")
    ap.add_argument("--default", default="runs/phaseE_ablation/results.jsonl")
    ap.add_argument("--metric", default="macro_f1")
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = []
    for d in Path(a.root).glob("*/results.jsonl"):
        m = re.match(r"(.+)_([0-9.]+)$", d.parent.name)
        if not m:
            continue
        df = load_results(d); df["param"], df["value"] = m.group(1), float(m.group(2))
        frames.append(df)
    if Path(a.default).exists():
        base = load_results(a.default)
        for p, v in DEFAULTS.items():
            b = base[base.method.isin(["validaug", "diffusion"])].copy(); b["param"], b["value"] = p, v
            frames.append(b)
    S = pd.concat(frames, ignore_index=True)
    out = Path(a.root) / "analysis"; out.mkdir(parents=True, exist_ok=True)
    tab = S.groupby(["param", "value", "method", "scenario"])[[a.metric, "rare_f1", "validity_raw", "c2st_auc"]].mean()
    tab.round(4).to_csv(out / "sensitivity.csv")
    for p, g in S.groupby("param"):
        fig, ax = plt.subplots(figsize=(4, 3))
        for (meth, scen), h in g.groupby(["method", "scenario"]):
            s = h.groupby("value")[a.metric].agg(["mean", "std"])
            ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=3, label=f"{meth}/{scen}")
        ax.set_xlabel(p); ax.set_ylabel(a.metric); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(out / f"sens_{p}.png", dpi=200); plt.close(fig)
    print("Written to", out)
