"""Registry of augmentation methods (baselines + VALID-Aug + ablations).

Every augmenter receives an ``AugContext`` and returns an ``AugResult``:
  Z_syn, y_syn : encoded synthetic samples (snapped to valid encodings)
  df_syn_raw   : samples in the original space BEFORE projection (true generator Validity)
  replace_train: True if the method replaces the whole training set (SMOTE-ENN cleaning)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .budget import AdaptiveBudget, default_budget
from .utils import LOG, timer


@dataclass
class AugContext:
    Z_tr: np.ndarray
    y_tr: np.ndarray
    Z_va: np.ndarray
    y_va: np.ndarray
    df_tr: pd.DataFrame  # original space (harmonized features)
    encoder: object
    constraints: object
    protocol_col: str | None
    seed: int
    device: object
    cfg: dict
    cache_dir: Path
    shared: dict = field(default_factory=dict)  # shared across variants (diffusion, boundary)


@dataclass
class AugResult:
    Z_syn: np.ndarray
    y_syn: np.ndarray
    df_syn_raw: pd.DataFrame | None = None
    replace_train: bool = False
    Z_full: np.ndarray | None = None
    y_full: np.ndarray | None = None
    info: dict = field(default_factory=dict)


def _empty(ctx):
    return AugResult(np.zeros((0, ctx.Z_tr.shape[1]), np.float32), np.array([], dtype=object))


def _budget(ctx):
    b = ctx.cfg.get("budget", {})
    return default_budget(ctx.y_tr, b.get("alpha", 0.5), b.get("beta", 10.0))


# ------------------------------------------------------------ baselines
def aug_none(ctx):
    return _empty(ctx)


def _imblearn(ctx, sampler_name):
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler

    bud = _budget(ctx)
    cnt = pd.Series(ctx.y_tr).value_counts()
    strat = {c: int(cnt[c] + n) for c, n in bud.items() if n > 0}
    if not strat:
        return _empty(ctx)
    kmin = int(min(cnt[c] for c in strat))
    k = max(1, min(5, kmin - 1))
    mk = {
        "ros": lambda: RandomOverSampler(sampling_strategy=strat, random_state=ctx.seed),
        "smote": lambda: SMOTE(sampling_strategy=strat, k_neighbors=k, random_state=ctx.seed),
        "borderline_smote": lambda: BorderlineSMOTE(sampling_strategy=strat, k_neighbors=k, random_state=ctx.seed),
        "adasyn": lambda: ADASYN(sampling_strategy=strat, n_neighbors=k, random_state=ctx.seed),
        "smote_enn": lambda: SMOTEENN(sampling_strategy=strat, random_state=ctx.seed,
                                      smote=SMOTE(sampling_strategy=strat, k_neighbors=k, random_state=ctx.seed)),
    }
    try:
        X, y = mk[sampler_name]().fit_resample(ctx.Z_tr, ctx.y_tr)
    except (ValueError, RuntimeError) as e:  # ADASYN may fail when no other-class neighbours exist
        LOG.warning("%s failed (%s) -> falling back to SMOTE", sampler_name, e)
        X, y = mk["smote"]().fit_resample(ctx.Z_tr, ctx.y_tr)
        sampler_name = "smote"
    if sampler_name == "smote_enn":
        df_raw = ctx.encoder.inverse_transform(X)
        return AugResult(np.zeros((0, X.shape[1]), np.float32), np.array([], dtype=object), df_raw,
                         replace_train=True, Z_full=ctx.encoder.snap(X), y_full=np.asarray(y, dtype=object),
                         info={"n_after": int(len(y))})
    Xs, ys = X[len(ctx.Z_tr):], np.asarray(y[len(ctx.Z_tr):], dtype=object)
    df_raw = ctx.encoder.inverse_transform(Xs)
    return AugResult(ctx.encoder.snap(Xs), ys, df_raw)


def _sdv_like(ctx, kind):
    """CTGAN / TVAE (`ctgan` package) in the original space, conditioned on the label."""
    try:
        from ctgan import CTGAN, TVAE
    except ImportError:
        LOG.warning("Package `ctgan` not installed -> skipping %s (pip install ctgan)", kind)
        return None
    enc = ctx.encoder
    df = ctx.df_tr[enc.num_cols + enc.cat_cols + enc.bit_cols].copy()
    # CTGAN/TVAE reject NaN: fill with the TRAIN median (same value as the encoder), normalize dtypes
    df[enc.num_cols] = df[enc.num_cols].astype("float64").fillna(pd.Series(enc.median, index=enc.num_cols))
    for c in enc.cat_cols:
        df[c] = df[c].astype(str)
    for c in enc.bit_cols:
        df[c] = df[c].fillna(0).astype("int64")
    df["__y"] = np.asarray(ctx.y_tr, dtype=str)
    disc = enc.cat_cols + enc.bit_cols + ["__y"]
    gcfg = ctx.cfg.get("gan", {})
    # CTGAN/TVAE are very slow on hundreds of thousands of rows -> train on a subsample (rare classes kept)
    max_rows = gcfg.get("max_rows", 50000)
    if max_rows and len(df) > max_rows:
        from .runner import cap_train

        keep = cap_train(np.arange(len(df)), df["__y"].to_numpy(object), max_rows, ctx.seed)
        df = df.iloc[keep].reset_index(drop=True)
        LOG.info("%s: training on a %d-row subsample (rare classes kept)", kind, len(df))
    gpu = str(ctx.device).startswith("cuda")
    Cls = CTGAN if kind == "ctgan" else TVAE
    kw = dict(epochs=gcfg.get("epochs", 300), batch_size=gcfg.get("batch_size", 500))
    try:
        model = Cls(**kw, enable_gpu=gpu)
    except TypeError:  # older ctgan versions
        model = Cls(**kw, cuda=gpu)
    import logging

    logging.getLogger("rdt").setLevel(logging.WARNING)
    model.fit(df, discrete_columns=disc)
    parts = []
    for c, n in _budget(ctx).items():
        if n > 0:
            if kind == "ctgan":
                s = model.sample(n, condition_column="__y", condition_value=c)
            else:  # TVAE is unconditional -> rejection sampling
                s, tries = [], 0
                while sum(len(x) for x in s) < n and tries < 50:
                    x = model.sample(max(n * 5, 1000)); s.append(x[x["__y"] == c]); tries += 1
                s = pd.concat(s).head(n)
            parts.append(s)
    if not parts:
        return _empty(ctx)
    raw = pd.concat(parts, ignore_index=True)
    y = raw.pop("__y").to_numpy(dtype=object)
    for c in enc.cat_cols:
        raw[c] = raw[c].astype(str)
    # Validity is reported on the DECODED output (integer columns rounded, values clipped to the train
    # range) -- exactly what the diffusion family reports as "raw" -- so validity_raw is comparable
    # across generators. The float output of CTGAN/TVAE (e.g. 3.2 packets) is not a flow.
    raw = enc.decode_like_generator(raw)
    return AugResult(enc.transform(raw), y, raw)


def _external(ctx, kind):
    """TabSyn / TabDDPM from their official code (separate environment), same budget as all baselines."""
    from .external import ExternalGenerator

    ecfg = ctx.cfg.get("external", {})
    if not ecfg.get("tabsyn_root"):
        LOG.warning("external.tabsyn_root not configured -> skipping %s (see scripts/external/setup_external.sh)", kind)
        return None
    enc = ctx.encoder
    quotas = {c: n for c, n in _budget(ctx).items() if n > 0}
    if not quotas:
        return _empty(ctx)
    gen = ExternalGenerator(kind, ecfg, ctx.cache_dir, ctx.seed)
    raw, info = gen.generate(ctx.df_tr, ctx.y_tr, enc.num_cols, enc.cat_cols + enc.bit_cols, quotas)
    if raw.empty:
        return AugResult(np.zeros((0, ctx.Z_tr.shape[1]), np.float32), np.array([], dtype=object), info=info)
    y = raw.pop("label").to_numpy(dtype=object)
    raw[enc.num_cols] = raw[enc.num_cols].apply(pd.to_numeric, errors="coerce")
    for c in enc.bit_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0).round().astype("int64")
    for c in enc.cat_cols:
        raw[c] = raw[c].astype(str)
    raw = enc.decode_like_generator(enc.clean(raw))  # same decoding as the diffusion family (see _sdv_like)
    return AugResult(enc.transform(raw), y, raw, info=info)


# ------------------------------------------------------------ diffusion
def _get_diffusion(ctx, fit_rows=None):
    """fit_rows: train the diffusion model on a class-capped subsample of that many rows (the same
    water-filling subsample rule as CTGAN/TVAE); None = all training rows (default)."""
    import torch

    from .diffusion import DiffusionConfig, TabDiffusion

    Z_fit, y_fit, skey = ctx.Z_tr, ctx.y_tr, "diffusion"
    if fit_rows and len(ctx.Z_tr) > fit_rows:
        from .runner import cap_train

        keep = cap_train(np.arange(len(ctx.y_tr)), np.asarray(ctx.y_tr, dtype=object), int(fit_rows), ctx.seed)
        Z_fit, y_fit, skey = ctx.Z_tr[keep], np.asarray(ctx.y_tr, dtype=object)[keep], f"diffusion_{int(fit_rows)}"
        LOG.info("diffusion: training on a %d-row subsample (rare classes kept)", len(keep))
    if skey in ctx.shared:
        return ctx.shared[skey]
    classes = sorted(set(ctx.y_tr))
    cfg = DiffusionConfig(**ctx.cfg.get("diffusion", {}))
    # The cache key fingerprints everything the model depends on (training data, classes, diffusion
    # hyper-parameters), so phases can share one cache safely and a changed setting forces retraining.
    import hashlib
    import json as _json

    fp = hashlib.sha1()
    fp.update(np.ascontiguousarray(Z_fit, dtype=np.float32).tobytes())
    fp.update(_json.dumps({"classes": classes, "y": pd.Series(y_fit).value_counts().sort_index().to_dict(),
                           "cfg": cfg.__dict__, "seed": ctx.seed}, sort_keys=True, default=str).encode())
    path = ctx.cache_dir / f"diffusion_{fp.hexdigest()[:12]}.pt"
    if path.exists():
        st = torch.load(path, map_location=ctx.device, weights_only=False)
        LOG.info("Loading diffusion model from cache %s", path)
        ctx.shared[skey] = TabDiffusion.from_state(st, ctx.device)
        return ctx.shared[skey]
    dm = TabDiffusion(ctx.Z_tr.shape[1], classes, cfg, ctx.device).fit(Z_fit, y_fit)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dm.state_dict(), path)
    ctx.shared[skey] = dm
    return dm


def _get_boundary(ctx):
    from .guidance import BoundaryEnsemble

    if "boundary" not in ctx.shared:
        bc = ctx.cfg.get("boundary", {})
        ctx.shared["boundary"] = BoundaryEnsemble(ctx.Z_tr.shape[1], sorted(set(ctx.y_tr)), device=ctx.device,
                                                  seed=ctx.seed, **bc).fit(ctx.Z_tr, ctx.y_tr)
    return ctx.shared["boundary"]


def aug_diffusion_family(ctx, variant: dict):
    """variant: guidance_c, guidance_b, projection, filter, budget('default'|'adaptive')."""
    from .diffusion import Guidance
    from .filtering import TripleFilter

    vc = ctx.cfg.get("validaug", {})
    t = {}
    with timer(t, "train_generator_s"):
        fit_rows = ctx.cfg.get("gan", {}).get("max_rows", 50000) if variant.get("fit_rows") == "gan" else None
        dm = _get_diffusion(ctx, fit_rows)
        bnd = _get_boundary(ctx) if (variant["guidance_b"] or variant["filter"]) else None
    # delta is shared by the boundary reward B(x,y) = -(m_y - delta)^2 AND the filter ranking score
    # -|m_y - delta|, so the `va_no_delta` ablation sets both to 0.
    delta = vc.get("delta", 1.0) if variant.get("delta", True) else 0.0
    guide = None
    if variant["guidance_c"] or variant["guidance_b"]:
        guide = Guidance(ctx.encoder, ctx.constraints if variant["guidance_c"] else None,
                         bnd if variant["guidance_b"] else None,
                         lambda_c=vc.get("lambda_c", 1.0), lambda_b=vc.get("lambda_b", 0.2),
                         gamma=vc.get("gamma", 1.0), t1=vc.get("t1", 0.2), t2=vc.get("t2", 0.7),
                         delta=delta, max_grad_norm=vc.get("max_grad_norm", 1.0),
                         inner_steps=vc.get("inner_steps", 3))
    bud = _budget(ctx)
    beta = ctx.cfg.get("budget", {}).get("beta", 10.0)
    cnt = pd.Series(ctx.y_tr).value_counts()
    cap = vc.get("pool_cap", 20000)  # caps the number of guided samples per class in BOTH modes
    if variant["budget"] == "adaptive":  # search pool: n_c <= min(beta*N_c, pool_cap)
        want = {c: int(min(beta * cnt[c], cap)) if cap else int(beta * cnt[c]) for c, n in bud.items() if n > 0}
    else:  # default budget (same as every baseline), capped by pool_cap (matters for tune_guidance)
        want = {c: int(min(n, cap)) if cap else n for c, n in bud.items() if n > 0}
    oversample = vc.get("oversample", 2.0) if variant["filter"] else 1.0
    Zs, ys, raws = [], [], []
    with timer(t, "generate_s"):
        for c, n in want.items():
            m = int(np.ceil(n * oversample))
            if m <= 0:
                continue
            Z = dm.sample(c, m, guide, batch=vc.get("sample_batch", 4096), seed=ctx.seed + zlib.crc32(str(c).encode()) % 1000)
            raw = ctx.encoder.inverse_transform(Z)
            raws.append(raw)
            if variant["projection"]:
                raw = ctx.constraints.repair(raw, ctx.encoder)
            Zs.append(ctx.encoder.transform(raw)); ys.append(np.full(m, c, dtype=object))
    if not Zs:
        return _empty(ctx)
    Z, y, raw0 = np.vstack(Zs), np.concatenate(ys), pd.concat(raws, ignore_index=True)
    info = dict(t)
    if guide is not None:
        info["guidance_V_mean"] = float(np.mean(guide.stats["V"])) if guide.stats["V"] else None
        info["guidance_B_mean"] = float(np.mean(guide.stats["B"])) if guide.stats["B"] else None
    keep = np.ones(len(y), bool)
    score = np.zeros(len(y))
    if variant["filter"]:
        fcfg = ctx.cfg.get("filter", {})
        with timer(info, "filter_s"):
            keep, score, frep = TripleFilter(seed=ctx.seed, **fcfg)(
                Z, y, ctx.Z_tr, ctx.y_tr, ctx.Z_va, ctx.y_va, bnd, delta)
        info["filter"] = frep
    pools = {}
    for c in want:
        ix = np.where((y == c) & keep)[0]
        ix = ix[np.argsort(-score[ix], kind="stable")] if variant["filter"] else ix
        pools[c] = ix
    if variant["budget"] == "adaptive":
        with timer(info, "budget_s"):
            ab = AdaptiveBudget(beta=beta, n_trials=vc.get("budget_trials", 30), seed=ctx.seed,
                                proxies=vc.get("budget_proxies", ("xgboost", "mlp")),
                                min_gain=vc.get("budget_min_gain", 0.002), device=ctx.device)
            n_sel, binfo = ab.search(ctx.Z_tr, ctx.y_tr, ctx.Z_va, ctx.y_va, {c: Z[ix] for c, ix in pools.items()})
        info["budget"] = binfo
    else:
        n_sel = {c: min(len(ix), bud[c]) for c, ix in pools.items()}
    sel = np.concatenate([pools[c][: n_sel.get(c, 0)] for c in pools]) if pools else np.array([], int)
    sel = sel.astype(int)
    info["n_selected"] = {c: int(n_sel.get(c, 0)) for c in pools}
    return AugResult(Z[sel], y[sel], raw0, info=info)


DIFFUSION_VARIANTS = {
    # name: (guidance_c, guidance_b, projection, filter, budget)
    "diffusion": dict(guidance_c=False, guidance_b=False, projection=False, filter=False, budget="default"),
    "diffusion_proj": dict(guidance_c=False, guidance_b=False, projection=True, filter=False, budget="default"),
    "validaug": dict(guidance_c=True, guidance_b=True, projection=True, filter=True, budget="adaptive"),
    # ---- control: the unguided diffusion model trained on the SAME class-capped subsample size as
    #      CTGAN/TVAE (aug.gan.max_rows), to separate model family from training-set size ----
    "diffusion_50k": dict(guidance_c=False, guidance_b=False, projection=False, filter=False, budget="default",
                          fit_rows="gan"),
    # ---- ablation ----
    "va_no_gc": dict(guidance_c=False, guidance_b=True, projection=True, filter=True, budget="adaptive"),
    "va_no_gb": dict(guidance_c=True, guidance_b=False, projection=True, filter=True, budget="adaptive"),
    "va_no_proj": dict(guidance_c=True, guidance_b=True, projection=False, filter=True, budget="adaptive"),
    "va_no_filter": dict(guidance_c=True, guidance_b=True, projection=True, filter=False, budget="adaptive"),
    "va_default_budget": dict(guidance_c=True, guidance_b=True, projection=True, filter=True, budget="default"),
    "va_no_delta": dict(guidance_c=True, guidance_b=True, projection=True, filter=True, budget="adaptive", delta=False),
    "va_guidance_only": dict(guidance_c=True, guidance_b=True, projection=False, filter=False, budget="default"),
}

METHODS = {
    "none": aug_none,
    "ros": lambda c: _imblearn(c, "ros"),
    "smote": lambda c: _imblearn(c, "smote"),
    "borderline_smote": lambda c: _imblearn(c, "borderline_smote"),
    "adasyn": lambda c: _imblearn(c, "adasyn"),
    "smote_enn": lambda c: _imblearn(c, "smote_enn"),
    "ctgan": lambda c: _sdv_like(c, "ctgan"),
    "tvae": lambda c: _sdv_like(c, "tvae"),
    "tabsyn": lambda c: _external(c, "tabsyn"),
    "tabddpm": lambda c: _external(c, "tabddpm"),
}
for _k, _v in DIFFUSION_VARIANTS.items():
    METHODS[_k] = (lambda v: (lambda c: aug_diffusion_family(c, v)))(_v)


def run_method(name: str, ctx: AugContext) -> AugResult | None:
    if name not in METHODS:
        raise KeyError(f"Unknown method: {name}. Available: {list(METHODS)}")
    t = {}
    with timer(t, "total_aug_s"):
        res = METHODS[name](ctx)
    if res is not None:
        res.info.update(t)
    return res
