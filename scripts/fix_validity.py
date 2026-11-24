"""Recompute `validity_raw` / `validity_raw_soft` of CTGAN / TVAE / TabSyn / TabDDPM rows that were produced
before the generators' raw output was decoded like the diffusion family (integers rounded, train range).

Reads runs/<phase>/results.jsonl, rebuilds encoder + constraints of each affected cell from the stored
split (same code path as the runner, so the fitted objects are identical), decodes the stored
synthetic/<cell>/<method>__raw.parquet sample and rewrites the two fields in place (a backup
results.jsonl.bak is kept). Rows whose raw sample is missing are left untouched and listed.

python scripts/fix_validity.py --exp configs/phaseD_main.yaml --results runs/phaseD_main/results.jsonl
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validaug.constraints import ConstraintSet  # noqa: E402
from validaug.encoder import TabularEncoder  # noqa: E402
from validaug.runner import _load, _types, cap_train  # noqa: E402
from validaug.schema import make_schema  # noqa: E402
from validaug.splits import apply_scenario, dedup_leakage  # noqa: E402
from validaug.utils import setup_logging  # noqa: E402

AFFECTED = ("ctgan", "tvae", "tabsyn", "tabddpm")


def fit_cell(cfg, dcfg, ds, split, scen_name, seed, out):
    scen = next((s for s in cfg.get("scenarios", []) if s["name"] == scen_name), {"type": "full"})
    df, rep = _load(cfg["data_dir"], ds)
    z = np.load(Path(out) / "splits" / f"{ds}__{split}__{0 if cfg.get('split_seed_fixed') else seed}.npz",
                allow_pickle=True)
    y_all = df["__y"].to_numpy(object)
    tr, va, _ = apply_scenario(z["train"], z["val"], y_all, scen, seed)
    if scen.get("task") == "binary":
        df = df.copy()
        df["__y"] = np.where(y_all == "Benign", "Benign", "Attack").astype(object)
        y_all = df["__y"].to_numpy(object)
        if split != "random_naive":
            tr, va, _, _ = dedup_leakage(df, tr, va, z["test"])
    cap = cfg.get("max_train_rows")
    if cap and len(tr) > cap:
        tr = cap_train(tr, y_all, cap, seed)
    num, cat, bit = _types(df, rep)
    fam = make_schema(dict(dcfg[ds], name=ds)).family
    dtr = df.iloc[tr].reset_index(drop=True)
    enc = TabularEncoder(**cfg.get("encoder", {})).fit(dtr, num, cat, bit)
    dtr = enc.clean(dtr)
    pcol = make_schema({"name": ds, "family": fam}).protocol_col
    cs = ConstraintSet(fam, **cfg.get("constraints", {})).fit(dtr, enc, pcol)
    return enc, cs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--datasets-config", default="configs/datasets.yaml")
    ap.add_argument("--methods", nargs="*", default=list(AFFECTED))
    a = ap.parse_args()
    setup_logging("WARNING")
    cfg = yaml.safe_load(open(a.exp)); dcfg = yaml.safe_load(open(a.datasets_config))["datasets"]
    res_path = Path(a.results); out = res_path.parent
    rows = [json.loads(l) for l in open(res_path, encoding="utf-8")]
    shutil.copy(res_path, res_path.with_suffix(".jsonl.bak"))
    cache, missing, changed = {}, [], 0
    for r in rows:
        if r["method"] not in a.methods or r["split"] == "cross":
            continue
        cell = f"{r['dataset']}__{r['split']}__{r['scenario']}__seed{r['seed']}"
        raw_p = out / "synthetic" / cell / f"{r['method']}__raw.parquet"
        if not raw_p.exists():
            missing.append((cell, r["method"])); continue
        if cell not in cache:
            cache[cell] = fit_cell(cfg, dcfg, r["dataset"], r["split"], r["scenario"], r["seed"], out)
        enc, cs = cache[cell]
        raw = pd.read_parquet(raw_p)
        for c in enc.cat_cols:
            raw[c] = raw[c].astype(str)
        dec = enc.decode_like_generator(enc.clean(raw))
        old = r.get("validity_raw")
        r["validity_raw_unrounded"] = old
        r["validity_raw"] = cs.violation_report(dec)["any"]
        r["validity_raw_soft"] = cs.violation_report(dec, include_soft=True)["any"]
        r["validity_raw_note"] = f"recomputed on {len(dec)} stored raw rows (decoded like the diffusion family)"
        changed += 1
        print(f"{cell:60s} {r['method']:8s} validity_raw {old} -> {r['validity_raw']:.4f}")
    with open(res_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"# updated {changed} rows; backup at {res_path.with_suffix('.jsonl.bak')}")
    if missing:
        print("# rows without a stored raw sample (left unchanged):", sorted(set(missing)))
