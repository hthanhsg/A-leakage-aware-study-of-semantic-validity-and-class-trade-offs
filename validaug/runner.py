"""Experiment-grid runner (resumable: interrupted runs continue from the result keys).

One result row = (dataset, split, scenario, seed, method, classifier) + metrics on the 4 axes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .augmenters import AugContext, run_method
from .budget import default_budget
from .classifiers import make_classifier
from .constraints import ConstraintSet
from .encoder import TabularEncoder
from .io import feature_columns
from .metrics import fidelity, privacy, utility
from .schema import make_schema
from .splits import apply_scenario, dedup_leakage, make_split, map_label_group
from .utils import (LOG, GPUEnergyMeter, append_jsonl, dump_json, get_device, set_seed, timer,
                    write_env_snapshot)

KEY = ("dataset", "split", "scenario", "seed", "method", "classifier")


def split_method(name: str):
    """Control baselines without generation: `<method>_cw` / `<method>_cw_sqrt` run <method> and train the
    classifier with class weights ("balanced" / "sqrt"). Returns (augmentation method, class_weight mode)."""
    for suffix, mode in (("_cw_sqrt", "sqrt"), ("_cw", "balanced")):
        if name.endswith(suffix):
            return name[: -len(suffix)], mode
    return name, None


def _load(data_dir, name):
    d = Path(data_dir) / name
    df = pd.read_parquet(d / "data.parquet")
    rep = json.load(open(d / "audit.json"))
    return df, rep


def _types(df, rep):
    cat = [c for c in rep["categorical"] if c in df.columns]
    bit = [c for c in rep["bit_features"] if c in df.columns]
    num = [c for c in feature_columns(df) if c not in cat + bit]
    return num, cat, bit


def tail_classes(y, benign="Benign", share=0.02):
    cnt = pd.Series(y).value_counts()
    att = cnt[cnt.index != benign]
    if att.empty:
        return []
    tail = att[att / len(y) < share].index.tolist()
    return tail or [att.idxmin()]


def cap_train(tr, y_all, cap, seed):
    """Reduce train to `cap` rows by water-filling: only large classes are cut, rare classes are kept."""
    rng = np.random.default_rng(seed)
    groups = pd.Series(tr).groupby(y_all[tr]).apply(lambda s: s.to_numpy())
    order_ = sorted(groups.index, key=lambda c: len(groups[c]))
    remaining, keep = cap, []
    for i, c in enumerate(order_):
        quota = remaining // (len(order_) - i)
        ix = groups[c]
        take = ix if len(ix) <= quota else rng.choice(ix, quota, replace=False)
        keep.append(take); remaining -= len(take)
    return np.sort(np.concatenate(keep))


class Runner:
    def __init__(self, cfg: dict, ds_cfg: dict):
        self.cfg, self.ds_cfg = cfg, ds_cfg
        self.out = Path(cfg["out_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.res_path = self.out / "results.jsonl"
        self.device = get_device(cfg.get("device", "auto"))
        self.done = set()
        if self.res_path.exists():
            for line in open(self.res_path):
                r = json.loads(line)
                self.done.add(tuple(str(r[k]) for k in KEY))
        LOG.info("Device: %s ; %d results already present (resume)", self.device, len(self.done))
        dump_json(cfg, self.out / "config_used.json")
        write_env_snapshot(self.out)

    # ------------------------------------------------------------ main
    def run(self):
        for seed in self.cfg["seeds"]:
            for ds in self.cfg.get("datasets", []):
                df, rep = _load(self.cfg["data_dir"], ds)
                for split in self.cfg["splits"]:
                    need_t = split.startswith("temporal")
                    if (need_t and not rep.get("has_time", True)) or (split == "group" and not rep.get("has_host", True)):
                        LOG.warning("[%s] skipping split %s: dataset has no %s column", ds, split,
                                    "timestamp" if need_t else "source-host")
                        continue
                    sp = self._split(df, ds, split, seed)
                    for scen in self.cfg.get("scenarios", [{"name": "full", "type": "full"}]):
                        if scen.get("datasets") and ds not in scen["datasets"]:
                            continue
                        if scen.get("splits") and split not in scen["splits"]:
                            continue
                        self._cell(ds, df, rep, split, sp, scen, seed)
            for pair in self.cfg.get("cross_dataset", []):
                self._cross(pair, seed)

    def _split(self, df, ds, split, seed):
        sseed = 0 if self.cfg.get("split_seed_fixed", False) else seed
        p = self.out / "splits" / f"{ds}__{split}__{sseed}.npz"
        if p.exists():
            z = np.load(p, allow_pickle=True)
            return {"train": z["train"], "val": z["val"], "test": z["test"], "info": z["info"].item()}
        sp = make_split(df, split, sseed, train_conflicts=self.cfg.get("train_conflicts", "majority"))
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, train=sp["train"], val=sp["val"], test=sp["test"], info=np.array(sp["info"], dtype=object))
        dump_json(sp["info"], p.with_suffix(".json"))
        LOG.info("[%s/%s/seed%d] split sizes (train, val, test): %s", ds, split, sseed, sp["info"]["sizes"])
        return sp

    # ------------------------------------------------------------ cell
    def _cell(self, ds, df, rep, split, sp, scen, seed):
        y_all = df["__y"].to_numpy(object)
        tr, va, sinfo = apply_scenario(sp["train"], sp["val"], y_all, scen, seed)
        if tr is None:
            LOG.warning("[%s] skipping scenario %s: %s", ds, scen["name"], sinfo["skip"])
            return
        te = sp["test"]
        if scen.get("task") == "binary":
            # Binary task on the SAME split: relabel to Benign/Attack, then re-run duplicate removal,
            # because test flows that differed from a training flow only by attack type become exact
            # duplicates once labels are merged.
            df = df.copy()
            df["__y"] = np.where(df["__y"].to_numpy(object) == "Benign", "Benign", "Attack").astype(object)
            y_all = df["__y"].to_numpy(object)
            if split != "random_naive":  # the naive protocol deliberately keeps duplicates
                tr, va, te, dinfo = dedup_leakage(df, tr, va, te)
                LOG.info("[%s|%s|%s] binary relabelling: extra duplicates removed %s", ds, split, scen["name"],
                         dinfo["dedup_removed"])
        cap = self.cfg.get("max_train_rows")
        if cap and len(tr) > cap:
            tr = cap_train(tr, y_all, cap, seed)
        fam = make_schema(dict(self.ds_cfg[ds], name=ds)).family
        rare = sinfo.get("rare_classes") or tail_classes(y_all[tr])
        if scen.get("task") == "binary":
            rare = [c for c in ("Attack", "Benign") if c in set(y_all[tr])]
            rare = [min(rare, key=lambda c: int((y_all[tr] == c).sum()))]  # minority side
        self._run_methods(ds, split, scen["name"], seed, df, rep, tr, va, te, y_all, fam,
                          rare=rare, holdout=sinfo.get("holdout"),
                          methods=scen.get("methods"), classifiers=scen.get("classifiers"),
                          exclude_methods=scen.get("exclude_methods"))

    def _run_methods(self, ds, split, scen, seed, df, rep, tr, va, te, y_all, family, rare, holdout,
                     df_test=None, y_test=None, methods=None, classifiers=None, exclude_methods=None):
        methods = [m for m in self.cfg["methods"] if not methods or m in methods]
        methods = [m for m in methods if m not in (exclude_methods or [])]
        classifiers = [c for c in self.cfg["classifiers"] if not classifiers or c in classifiers]
        # If the shared budget policy gives 0 synthetic samples to every class (e.g. a binary task where
        # attacks are >= alpha * benign), every augmenter is a no-op and would only waste hours of
        # generator training: run `none` only and say so.
        bcfg = self.cfg.get("aug", {}).get("budget", {})
        bud = default_budget(y_all[tr], bcfg.get("alpha", 0.5), bcfg.get("beta", 10.0))
        if not any(v > 0 for v in bud.values()):
            LOG.warning("[%s|%s|%s|s%d] budget policy gives 0 samples to every class (%s) -> running `none` only",
                        ds, split, scen, seed, {k: int(v) for k, v in pd.Series(y_all[tr]).value_counts().items()})
            methods = [m for m in methods if split_method(m)[0] == "none"]
        todo = [(m, c) for m in methods for c in classifiers
                if (ds, split, scen, str(seed), m, c) not in self.done]
        if not todo:
            return
        set_seed(seed)
        num, cat, bit = _types(df, rep)
        # Validation rows of classes absent from train (possible after train-conflict resolution) are
        # dropped once here: every consumer (early stopping, filter, budget) indexes val by train classes.
        va = va[np.isin(y_all[va], np.unique(y_all[tr]))]
        dtr, dva = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
        dte = df.iloc[te].reset_index(drop=True) if df_test is None else df_test
        yte = y_all[te] if y_test is None else y_test
        enc = TabularEncoder(**self.cfg.get("encoder", {})).fit(dtr, num, cat, bit)
        dtr, dva, dte = enc.clean(dtr), enc.clean(dva), enc.clean(dte)  # NaN/inf from train statistics
        pcol = make_schema({"name": ds, "family": family}).protocol_col
        cs = ConstraintSet(family, **self.cfg.get("constraints", {})).fit(dtr, enc, pcol)
        Ztr, Zva, Zte = enc.transform(dtr), enc.transform(dva), enc.transform(dte)
        cell = f"{ds}__{split}__{scen}__seed{seed}".replace("->", "_to_")
        cdesc = cs.describe()
        cdesc["real_test_violation"] = cs.violation_report(dte)
        cdesc["encoder"] = {"num_cols": enc.num_cols, "cat_cols": enc.cat_cols, "bit_cols": enc.bit_cols,
                            "categories": enc.cats, "dim": enc.dim}
        dump_json(cdesc, self.out / "constraints" / f"{cell}.json")
        ytr, yva = y_all[tr], y_all[va]
        classes = sorted(set(ytr))
        base = {"dataset": ds, "split": split, "scenario": scen, "seed": seed, "n_train_real": int(len(tr)),
                "n_val": int(len(va)), "n_test": int(len(dte)), "dim": enc.dim,
                "n_hard_constraints": len(cs.hard), "n_soft_constraints": len(cs.soft),
                "real_test_violation": cs.violation_report(dte)["any"]}
        cache = Path(self.cfg.get("cache_root") or self.out / "cache") / ds / split / scen / f"seed{seed}"
        ctx = AugContext(Ztr, ytr, Zva, yva, dtr, enc, cs, pcol, seed, self.device, aug_config(self.cfg, ds), cache)
        for method in dict.fromkeys(m for m, _ in todo):
            set_seed(seed)
            t = {}
            with GPUEnergyMeter() as em, timer(t, "aug_wall_s"):
                aug_name, cw_mode = split_method(method)
                res = run_method(aug_name, ctx)
            if res is None:
                LOG.warning("Skipping %s (not available)", method)
                continue
            am = {"aug_energy_j": em.energy_j, **t, "aug_info": res.info}
            if res.replace_train:
                Zfit, yfit = res.Z_full, res.y_full
                am["n_syn"] = int(len(yfit) - len(ytr))
            else:
                Zfit, yfit = np.vstack([Ztr, res.Z_syn]), np.concatenate([ytr, res.y_syn])
                am["n_syn"] = int(len(res.y_syn))
                am["n_syn_per_class"] = pd.Series(res.y_syn).value_counts().to_dict()
            if res.df_syn_raw is not None and len(res.df_syn_raw):
                am["validity_raw"] = cs.violation_report(res.df_syn_raw)["any"]
                am["validity_raw_soft"] = cs.violation_report(res.df_syn_raw, include_soft=True)["any"]
            if len(res.y_syn):
                am["validity_final"] = cs.violation_report(enc.inverse_transform(res.Z_syn))["any"]
                if self.cfg.get("eval", {}).get("fidelity", True):
                    am.update(fidelity(Ztr, ytr, res.Z_syn, res.y_syn, len(enc.num_cols), seed))
                if self.cfg.get("eval", {}).get("privacy", True):
                    am.update(privacy(Ztr, Zva, res.Z_syn, seed))
            self._save_synthetic(cell, method, res, enc)
            for clf in [c for m, c in todo if m == method]:
                set_seed(seed)
                tt = {}
                params = dict(self.cfg.get("classifier_params", {}).get(clf, {}))
                if cw_mode:
                    params["class_weight"] = cw_mode
                with GPUEnergyMeter() as em2, timer(tt, "train_s"):
                    model = make_classifier(clf, classes, seed, self.device, params).fit(Zfit, yfit, Zva, yva)
                with timer(tt, "infer_s"):
                    P = model.predict_proba(Zte)
                u = utility(yte, P, classes, rare=rare, holdout=holdout)
                row = {**base, "method": method, "classifier": clf, **u, **am, **tt,
                       "train_energy_j": em2.energy_j, "rare_classes": rare, "timestamp": time.time()}
                append_jsonl(row, self.res_path)
                self.done.add((ds, split, scen, str(seed), method, clf))
                LOG.info("[%s|%s|%s|s%d] %-18s %-14s macroF1=%.4f rareF1=%s viol=%s",
                         ds, split, scen, seed, method, clf, u.get("macro_f1", np.nan),
                         _f(u.get("rare_f1")), _f(am.get("validity_raw")))

    def _save_synthetic(self, cell, method, res, enc):
        """Save a capped sample of synthetic flows (original space) for inspection and figures."""
        n = int(self.cfg.get("save_synthetic", 10000) or 0)
        if n <= 0 or not len(res.y_syn):
            return
        d = self.out / "synthetic" / cell
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        ix = rng.choice(len(res.y_syn), min(n, len(res.y_syn)), replace=False)
        fin = enc.inverse_transform(res.Z_syn[ix]); fin["__y"] = res.y_syn[ix]
        fin.to_parquet(d / f"{method}.parquet", index=False)
        if res.df_syn_raw is not None and len(res.df_syn_raw):  # generator output before projection
            raw = res.df_syn_raw.sample(min(n, len(res.df_syn_raw)), random_state=0)
            raw.to_parquet(d / f"{method}__raw.parquet", index=False)

    # ------------------------------------------------------- cross-dataset
    def _cross(self, pair, seed):
        A, B, mode = pair["train"], pair["test"], pair.get("labels", "binary")
        dfa, repa = _load(self.cfg["data_dir"], A)
        dfb, repb = _load(self.cfg["data_dir"], B)
        common = [c for c in feature_columns(dfa) if c in set(feature_columns(dfb))]
        cat = [c for c in repa["categorical"] if c in common]
        bit = [c for c in repa["bit_features"] if c in common]
        dfa, dfb = dfa[common + ["__y", "__time", "__host"]].copy(), dfb[common + ["__y", "__time", "__host"]].copy()
        groups = self.cfg.get("label_groups", {})
        for d in (dfa, dfb):
            d["__y"] = (np.where(d["__y"] == "Benign", "Benign", "Attack") if mode == "binary"
                        else map_label_group(d["__y"], groups))
        rep = {"categorical": cat, "bit_features": bit}
        sp = make_split(dfa, pair.get("split", "temporal_stratified"), seed,
                        train_conflicts=self.cfg.get("train_conflicts", "majority"))
        # deduplicate test (B) against train (A)
        both = pd.concat([dfa, dfb], ignore_index=True)
        n_a = len(dfa)
        trn, _, teb, _ = dedup_leakage(both, sp["train"], sp["val"], np.arange(n_a, len(both)))
        dte = both.iloc[teb].reset_index(drop=True)
        fam = make_schema(dict(self.ds_cfg[A], name=A)).family
        y_all = dfa["__y"].to_numpy(object)
        rare = tail_classes(y_all[trn]) if mode != "binary" else []
        self._run_methods(f"{A}->{B}", "cross", mode, seed, dfa, rep, trn, sp["val"], None, y_all, fam,
                          rare=rare, holdout=None, df_test=dte, y_test=dte["__y"].to_numpy(object))


def _f(x):
    return "na" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def aug_config(cfg: dict, ds: str) -> dict:
    """`aug` block with the per-dataset overrides of `aug_per_dataset` applied (phase-C guidance parameters
    differ per dataset). A cross-dataset cell "A->B" uses the overrides of its TRAINING dataset A."""
    base = cfg.get("aug", {})
    over = cfg.get("aug_per_dataset", {}) or {}
    key = ds.split("->")[0]
    return _deep_merge(base, over.get(key, {})) if key in over else base


def build_context(cfg: dict, ds_cfg: dict, ds: str, split: str, seed: int, scenario: dict | None = None):
    """Build the AugContext of one cell (used for guidance hyper-parameter tuning)."""
    df, rep = _load(cfg["data_dir"], ds)
    sp = make_split(df, split, seed, train_conflicts=cfg.get("train_conflicts", "majority"))
    y_all = df["__y"].to_numpy(object)
    tr, va, _ = apply_scenario(sp["train"], sp["val"], y_all, scenario or {"type": "full"}, seed)
    cap = cfg.get("max_train_rows")
    if cap and len(tr) > cap:  # same training rows as the runner -> same diffusion cache entry
        tr = cap_train(tr, y_all, cap, seed)
    va = va[np.isin(y_all[va], np.unique(y_all[tr]))]  # same rule as Runner._run_methods
    num, cat, bit = _types(df, rep)
    fam = make_schema(dict(ds_cfg[ds], name=ds)).family
    dtr, dva = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
    enc = TabularEncoder(**cfg.get("encoder", {})).fit(dtr, num, cat, bit)
    dtr, dva = enc.clean(dtr), enc.clean(dva)
    pcol = make_schema({"name": ds, "family": fam}).protocol_col
    cs = ConstraintSet(fam, **cfg.get("constraints", {})).fit(dtr, enc, pcol)
    cache = (Path(cfg.get("cache_root") or Path(cfg["out_dir"]) / "cache") / ds / split
             / (scenario or {"name": "full"}).get("name", "full") / f"seed{seed}")
    return AugContext(enc.transform(dtr), y_all[tr], enc.transform(dva), y_all[va], dtr, enc, cs, pcol, seed,
                      get_device(cfg.get("device", "auto")), aug_config(cfg, ds), cache)
