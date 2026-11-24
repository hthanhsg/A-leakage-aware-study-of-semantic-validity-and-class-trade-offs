"""Tune guidance hyper-parameters (lambda_c, lambda_b, delta, t1, t2) with Optuna on VAL (test is never touched).

python scripts/tune_guidance.py --exp configs/exp_full.yaml --dataset nf_unsw_v3 --split temporal_stratified --trials 30
Prints a YAML block to paste into aug.validaug of the config. The diffusion model is cached, so each
trial only costs sampling + filtering.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validaug.augmenters import aug_diffusion_family  # noqa: E402
from validaug.budget import _proxy_f1  # noqa: E402
from validaug.runner import aug_config, build_context  # noqa: E402
from validaug.utils import setup_logging  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--datasets-config", default="configs/datasets.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="temporal_stratified")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--set", nargs="*", default=[], help="override nested keys, e.g. aug.validaug.pool_cap=5000")
    ap.add_argument("--out", default="runs", help="where tune_<dataset>.yaml / .csv are written")
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0],
                    help="budget fractions evaluated per trial (objective = best)")
    a = ap.parse_args()
    setup_logging("WARNING")
    import optuna

    cfg = yaml.safe_load(open(a.exp)); dcfg = yaml.safe_load(open(a.datasets_config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        node = cfg
        *path, last = k.split(".")
        for p_ in path:
            node = node.setdefault(p_, {})
        node[last] = yaml.safe_load(v)
    ctx = build_context(cfg, dcfg["datasets"], a.dataset, a.split, a.seed)
    classes = sorted(set(ctx.y_tr))
    FRACTIONS = sorted(a.fractions)
    ok_va = np.isin(ctx.y_va, classes)
    aug_base = aug_config(cfg, a.dataset)
    base_vc = dict(aug_base.get("validaug", {}))
    proxies = tuple(base_vc.get("budget_proxies", ("xgboost", "mlp")))  # same proxy as the adaptive budget
    base_f1 = _proxy_f1(ctx.Z_tr, ctx.y_tr, ctx.Z_va[ok_va], ctx.y_va[ok_va], classes, a.seed, proxies, ctx.device)
    print(f"# reference: proxy {list(proxies)} val macro-F1 WITHOUT augmentation = {base_f1:.4f}", flush=True)
    # default budget, capped per class by aug.validaug.pool_cap (otherwise one trial may generate 100k+
    # guided samples per class); the search space is identical to phase D's `validaug`
    variant = dict(guidance_c=True, guidance_b=True, projection=True, filter=True, budget="default")

    def objective(trial):
        vc = dict(base_vc,
                  lambda_c=trial.suggest_float("lambda_c", 0.1, 2.0, log=True),
                  lambda_b=trial.suggest_float("lambda_b", 0.02, 1.0, log=True),
                  delta=trial.suggest_float("delta", 0.25, 4.0, log=True),
                  t1=trial.suggest_float("t1", 0.0, 0.4),
                  t2=trial.suggest_float("t2", 0.5, 0.95))
        ctx.cfg = dict(aug_base, validaug=vc)
        res = aug_diffusion_family(ctx, variant)
        # VALID-Aug never adds the full default budget blindly (its adaptive budget can add fewer samples),
        # so each trial is scored by the BEST of several budget fractions. Samples of each class are already
        # sorted by filter score, so a fraction f keeps the f*n_c best-scored samples of class c.
        f1s = {}
        for f in FRACTIONS:
            keep = []
            for c in np.unique(res.y_syn):
                ix = np.where(res.y_syn == c)[0]
                keep.append(ix[: int(np.ceil(f * len(ix)))])
            keep = np.concatenate(keep) if keep else np.array([], int)
            Z = np.vstack([ctx.Z_tr, res.Z_syn[keep]]); y = np.concatenate([ctx.y_tr, res.y_syn[keep]])
            f1s[f] = _proxy_f1(Z, y, ctx.Z_va[ok_va], ctx.y_va[ok_va], classes, a.seed, proxies, ctx.device)
        best_f = max(f1s, key=f1s.get)
        f1 = f1s[best_f]
        trial.set_user_attr("best_fraction", best_f)
        trial.set_user_attr("f1_by_fraction", {str(k): round(v, 5) for k, v in f1s.items()})
        trial.set_user_attr("n_syn_full_budget", int(len(res.y_syn)))
        trial.set_user_attr("gain_vs_no_aug", float(f1 - base_f1))
        print(f"# trial {trial.number}: F1 by budget fraction {dict((k, round(v, 4)) for k, v in f1s.items())} "
              f"(no aug {base_f1:.4f})", flush=True)
        trial.set_user_attr("generate_s", float(res.info.get("generate_s", np.nan)))
        trial.set_user_attr("filter_s", float(res.info.get("filter_s", np.nan)))
        return f1

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=a.seed))
    st.optimize(objective, n_trials=a.trials)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    st.trials_dataframe().to_csv(Path(a.out) / f"tune_{a.dataset}.csv", index=False)  # all trials (appendix)
    best = {"validaug": {**base_vc, **{k: round(v, 4) for k, v in st.best_params.items()}}}
    print(f"# best val macro-F1 = {st.best_value:.4f} (no augmentation {base_f1:.4f}, gain {st.best_value - base_f1:+.4f})")
    print(yaml.safe_dump(best))
    (Path(a.out) / f"tune_{a.dataset}.yaml").write_text(
        f"# best val macro-F1 {st.best_value:.4f}; no augmentation {base_f1:.4f}\n" + yaml.safe_dump(best))
