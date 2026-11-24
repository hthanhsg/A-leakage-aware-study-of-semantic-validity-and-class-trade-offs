"""Adapters for external generators run from their OFFICIAL code in a separate Python environment.

* ``tabsyn``  : TabSyn (Zhang et al., ICLR 2024) -- VAE + latent score-based diffusion.
* ``tabddpm`` : TabDDPM (Kotelnikov et al., ICML 2023) as shipped with the TabSyn repository.

Both are trained on the flows of the training partition (original feature space, label as a
categorical column) and are therefore *unconditional*; class quotas are met by rejection
sampling. Training uses a class-capped subsample (``max_rows``, rare classes kept whole, as for
CTGAN/TVAE) so that rare classes are frequent enough in the generated stream.

The external code is executed with ``external.python`` inside ``external.tabsyn_root``
(see scripts/external/setup_external.sh). Generated pools are cached by fingerprint, so every
classifier / phase re-uses the same synthetic data.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import LOG

TARGET = "label"
PFX = "v_"

TABDDPM_TOML = """parent_dir = "configs/{name}"
model_save_path = "ckpt/{name}"
sample_save_path = "sampled/{name}"
real_data_path = "data/{name}"
num_numerical_features = {n_num}
model_type = "mlp"
task_type = "{task}"

[model_params]
num_classes = {k}
is_y_cond = {y_cond}

[model_params.rtdl_params]
d_layers = [1024, 2048, 2048, 1024]
dropout = 0.0

[diffusion_params]
num_timesteps = 1000
gaussian_loss_type = "mse"

[train.main]
steps = {steps}
lr = 0.001809824563637657
weight_decay = 5e-4
batch_size = 4096

[train.T]
seed = {seed}
normalization = "quantile"
num_nan_policy = "mean"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"

[sample]
num_samples = {n_sample}
batch_size = {sample_batch}
seed = {seed}
"""


class ExternalGenerator:
    def __init__(self, kind: str, ecfg: dict, cache_dir: Path, seed: int):
        self.kind, self.cfg, self.seed = kind, ecfg, seed
        self.root = Path(os.path.expanduser(ecfg.get("tabsyn_root", "external/tabsyn"))).resolve()
        self.python = os.path.expanduser(ecfg.get("python") or "python")
        self.cache = Path(cache_dir).resolve() / "external"  # absolute: the external code runs in its repo
        if not (self.root / "main.py").exists():
            raise FileNotFoundError(f"TabSyn repository not found at {self.root}; "
                                    "run scripts/external/setup_external.sh and set external.tabsyn_root")

    # ------------------------------------------------------------ helpers
    def _run(self, args, log, env_extra=None):
        env = dict(os.environ, **(env_extra or {}))
        env["PYTHONPATH"] = str(self.root) + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [self.python] + args
        t0 = time.time()
        with open(log, "a") as f:
            f.write(f"\n$ {' '.join(cmd)}\n"); f.flush()
            r = subprocess.run(cmd, cwd=self.root, env=env, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            tail = Path(log).read_text().splitlines()[-30:]
            raise RuntimeError(f"{self.kind} command failed ({' '.join(args[:4])}...). Log tail:\n" + "\n".join(tail))
        return time.time() - t0

    def _write_dataset(self, name, df, num_cols, cat_cols):
        d = self.root / "data" / name
        d.mkdir(parents=True, exist_ok=True)
        cols = num_cols + cat_cols + [TARGET]
        df = df[cols]
        df.to_csv(d / f"{name}.csv", index=False)
        # "test" split required by the official preprocessing: a subset of the TRAINING rows
        # (only used for VAE monitoring; no information beyond train is exposed).
        df.sample(min(2000, len(df)), random_state=self.seed).to_csv(d / "test_src.csv", index=False)
        k = df[TARGET].nunique()
        info = {"name": name, "task_type": "multiclass" if k > 2 else "binclass", "header": "infer",
                "column_names": None, "num_col_idx": list(range(len(num_cols))),
                "cat_col_idx": list(range(len(num_cols), len(num_cols) + len(cat_cols))),
                "target_col_idx": [len(cols) - 1], "file_type": "csv",
                "data_path": f"data/{name}/{name}.csv", "test_path": f"data/{name}/test_src.csv"}
        (self.root / "data" / "Info").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "Info" / f"{name}.json").write_text(json.dumps(info, indent=2))
        return info, k

    # ------------------------------------------------------------ main entry
    def generate(self, df_tr: pd.DataFrame, y_tr: np.ndarray, num_cols, cat_cols, quotas: dict) -> tuple[pd.DataFrame, dict]:
        """Return generated rows (original space, incl. TARGET) satisfying `quotas` as far as possible."""
        from .runner import cap_train

        max_rows = int(self.cfg.get("max_rows", 50000))
        df = df_tr[num_cols + cat_cols].copy()
        # Categorical values get a non-numeric prefix: the official preprocessing re-reads the CSV and
        # would otherwise infer integer dtypes for codes such as protocol "6" (and fail).
        for c in cat_cols:
            df[c] = PFX + df[c].astype(str)
        df[TARGET] = PFX + np.asarray(y_tr).astype(str)
        if len(df) > max_rows:
            df = df.iloc[cap_train(np.arange(len(df)), df[TARGET].to_numpy(object), max_rows, self.seed)]
        df = df.reset_index(drop=True)
        # Classes not present in the subsample cannot be generated
        quotas = {PFX + str(c): int(n) for c, n in quotas.items() if n > 0}

        settings = {k: self.cfg.get(k) for k in ("tabsyn_vae_epochs", "tabsyn_epochs", "tabddpm_steps", "max_rows")}
        settings["tabddpm_y_cond"] = bool(self.cfg.get("tabddpm_y_cond", True))
        fp = hashlib.sha1(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
        fp.update(json.dumps([self.kind, settings, self.seed], sort_keys=True, default=str).encode())
        key = fp.hexdigest()[:10]
        name = f"va_{self.kind}_{key}"
        self.cache.mkdir(parents=True, exist_ok=True)
        pool_path, log = self.cache / f"{name}_pool.csv", self.cache / f"{name}.log"
        info = {"external_name": name, "train_rows": int(len(df)), "log": str(log)}

        pool = pd.read_csv(pool_path, dtype={c: str for c in cat_cols + [TARGET]}) if pool_path.exists() \
            else pd.DataFrame()
        if not pool.empty:
            LOG.info("%s: loaded cached synthetic pool %s (%d rows)", self.kind, pool_path, len(pool))
        have = pool[TARGET].value_counts().to_dict() if not pool.empty else {}
        need = {c: q for c, q in quotas.items() if have.get(c, 0) < q}

        if need:
            if not (self.root / "data" / name / "info.json").exists():
                self._write_dataset(name, df, num_cols, cat_cols)
                info["preprocess_s"] = self._run(["process_dataset.py", "--dataname", name], log)
            info.update(self._train(name, len(num_cols), df[TARGET].nunique(), len(df), log))
            parts, rounds = [pool] if not pool.empty else [], 0
            max_rounds = int(self.cfg.get("max_rounds", 20))
            t0 = time.time()
            while need and rounds < max_rounds:
                rounds += 1
                out = self.cache / f"{name}_round{rounds}.csv"
                self._sample(name, out, log)
                s = pd.read_csv(out, dtype={c: str for c in cat_cols + [TARGET]})
                out.unlink(missing_ok=True)
                s = s[s[TARGET].isin(list(need))]
                take = []
                for c, q in need.items():
                    sc = s[s[TARGET] == c]
                    take.append(sc.head(q - have.get(c, 0)))
                got = pd.concat(take, ignore_index=True) if take else pd.DataFrame()
                if len(got):
                    parts.append(got)
                    for c, n in got[TARGET].value_counts().items():
                        have[c] = have.get(c, 0) + int(n)
                need = {c: q for c, q in quotas.items() if have.get(c, 0) < q}
                LOG.info("%s: sampling round %d, still missing %s", self.kind, rounds,
                         {c.removeprefix(PFX): q - have.get(c, 0) for c, q in need.items()} or "nothing")
            pool = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=df.columns)
            pool.to_csv(pool_path, index=False)
            info.update(sample_s=time.time() - t0, sampling_rounds=rounds)
        info["shortfall"] = {c: int(q - have.get(c, 0)) for c, q in quotas.items() if have.get(c, 0) < q}
        if info["shortfall"]:
            LOG.warning("%s: quotas not met after rejection sampling: %s", self.kind,
                        {c.removeprefix(PFX): v for c, v in info["shortfall"].items()})
        # final selection per quota
        sel = [pool[pool[TARGET] == c].head(q) for c, q in quotas.items()]
        out = pd.concat(sel, ignore_index=True) if sel else pool.iloc[:0]
        out = out.copy()
        for c in cat_cols + [TARGET]:
            out[c] = out[c].astype(str).str.removeprefix(PFX)
        info["shortfall"] = {c.removeprefix(PFX): v for c, v in info["shortfall"].items()}
        if self.cfg.get("cleanup", False):
            for p in [self.root / "data" / name, self.root / "tabsyn" / "vae" / "ckpt" / name,
                      self.root / "tabsyn" / "ckpt" / name, self.root / "baselines" / "tabddpm" / "ckpt" / name]:
                shutil.rmtree(p, ignore_errors=True)
        return out, info

    def _train(self, name, n_num, k, n_rows, log) -> dict:
        t = {}
        if self.kind == "tabsyn":
            env = {"TABSYN_VAE_EPOCHS": str(self.cfg.get("tabsyn_vae_epochs", 4000)),
                   "TABSYN_EPOCHS": str(self.cfg.get("tabsyn_epochs", 10000))}
            if not (self.root / "tabsyn" / "vae" / "ckpt" / name / "train_z.npy").exists():
                t["train_vae_s"] = self._run(["main.py", "--dataname", name, "--method", "vae", "--mode", "train",
                                              "--gpu", "0"], log, env)
            if not (self.root / "tabsyn" / "ckpt" / name / "model.pt").exists():
                t["train_diffusion_s"] = self._run(["main.py", "--dataname", name, "--method", "tabsyn", "--mode",
                                                    "train", "--gpu", "0"], log, env)
        else:
            cfgp = self.root / "baselines" / "tabddpm" / "configs" / f"{name}.toml"
            # is_y_cond = true: the official class-conditional TabDDPM (the label conditions the denoiser);
            # the unconditional variant (false) is a weaker baseline than the method actually proposed.
            y_cond = "true" if self.cfg.get("tabddpm_y_cond", True) else "false"
            cfgp.write_text(TABDDPM_TOML.format(name=name, n_num=n_num, task="multiclass" if k > 2 else "binclass",
                                                k=k, y_cond=y_cond, steps=int(self.cfg.get("tabddpm_steps", 30000)),
                                                seed=self.seed, n_sample=int(self.cfg.get("tabddpm_sample_rows", n_rows)),
                                                sample_batch=min(10000, int(self.cfg.get("tabddpm_sample_rows", n_rows)))))
            if not (self.root / "baselines" / "tabddpm" / "ckpt" / name / "model.pt").exists():
                t["train_s"] = self._run(["main.py", "--dataname", name, "--method", "tabddpm", "--mode", "train",
                                          "--gpu", "0"], log)
        return t

    def _sample(self, name, out, log):
        args = ["main.py", "--dataname", name, "--method", self.kind, "--mode", "sample", "--gpu", "0",
                "--save_path", str(out)]
        if self.kind == "tabddpm" and self.cfg.get("tabddpm_ddim", False):  # faster, non-default sampler
            args += ["--ddim", "--steps", str(self.cfg.get("tabddpm_ddim_steps", 50))]
        self._run(args, log)
