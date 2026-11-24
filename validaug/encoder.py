"""Invertible encoder T (fitted on TRAIN only).

* Numerical features : median imputation -> signed-log -> quantile-normal (custom implementation
  with a differentiable torch counterpart, required by constraint guidance).
* Categorical features : top-K values + "__other__", one-hot on the {-1,+1} scale.
* Bits (split TCP flags) : {0,1} -> {-1,+1}.

``inverse_transform`` maps back to the original space (bytes/packets/ms units) to check
constraints; ``decode_torch`` is the differentiable version returning original numerical values
and soft probabilities for categorical/bit features.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

Z_CLIP = 5.2
P_EPS = 1e-7


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def islog(y):
    return np.sign(y) * np.expm1(np.abs(y))


@dataclass
class Block:
    name: str
    kind: str  # "num" | "cat" | "bit"
    start: int
    end: int
    categories: list = field(default_factory=list)


class TabularEncoder:
    def __init__(self, n_quantiles: int = 1000, max_categories: int = 32):
        self.n_quantiles = n_quantiles
        self.max_categories = max_categories

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, num_cols, cat_cols, bit_cols):
        self.num_cols, self.cat_cols, self.bit_cols = list(num_cols), list(cat_cols), list(bit_cols)
        X = df[self.num_cols].to_numpy(dtype="float64")
        F = np.where(np.isfinite(X), X, np.nan)
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.median = np.nan_to_num(np.nanmedian(F, axis=0))
                self.lo = np.nan_to_num(np.nanmin(F, axis=0))
                self.hi = np.nan_to_num(np.nanmax(F, axis=0))
        X = self._clean_arr(X)
        self.is_int = np.all(np.isclose(X, np.round(X)), axis=0)
        # Dequantize integer features + add a tiny ramp so that q is strictly increasing:
        # avoids flat regions of the inverse quantile map -> guidance gradients do not vanish.
        rng = np.random.default_rng(0)
        Xj = X + np.where(self.is_int, rng.uniform(-0.49, 0.49, X.shape), 0.0)
        S = slog(Xj)
        nq = int(min(self.n_quantiles, max(10, len(df))))
        self.probs = np.linspace(0, 1, nq)
        q = np.quantile(S, self.probs, axis=0).T  # (n_num, nq)
        span = np.maximum(q[:, -1:] - q[:, :1], 1.0)
        self.q = q + span * 1e-4 * np.linspace(0, 1, nq)[None, :]
        # scale for the constraint energy (original space)
        self.scale = np.maximum(np.quantile(np.abs(X), 0.99, axis=0), 1.0)
        self.cats = {}
        for c in self.cat_cols:
            vc = df[c].astype(str).value_counts()
            self.cats[c] = list(vc.index[: self.max_categories]) + ["__other__"]
        self._build_blocks()
        return self

    def _build_blocks(self):
        self.blocks: list[Block] = []
        p = 0
        for c in self.num_cols:
            self.blocks.append(Block(c, "num", p, p + 1)); p += 1
        for c in self.cat_cols:
            k = len(self.cats[c])
            self.blocks.append(Block(c, "cat", p, p + k, self.cats[c])); p += k
        for c in self.bit_cols:
            self.blocks.append(Block(c, "bit", p, p + 1)); p += 1
        self.dim = p
        self.num_idx = np.arange(len(self.num_cols))

    # ------------------------------------------------------------ transform
    def _clean_arr(self, X):
        """+inf -> finite train max, -inf -> min, NaN -> median (all from TRAIN)."""
        X = np.where(np.isposinf(X), self.hi, X)
        X = np.where(np.isneginf(X), self.lo, X)
        return np.where(np.isnan(X), self.median, X)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame copy without NaN/inf in numerical columns (used by constraints, CTGAN, Validity)."""
        out = df.copy()
        if self.num_cols:
            out[self.num_cols] = self._clean_arr(out[self.num_cols].to_numpy(dtype="float64"))
        return out

    def _num_fwd(self, X):
        X = self._clean_arr(X)
        S = slog(X)
        Z = np.empty_like(S)
        for j in range(S.shape[1]):
            q = self.q[j]
            a = np.interp(S[:, j], q, self.probs)
            b = 1.0 - np.interp(-S[:, j], -q[::-1], 1.0 - self.probs[::-1])
            Z[:, j] = ndtri(np.clip(0.5 * (a + b), P_EPS, 1 - P_EPS))
        return np.clip(Z, -Z_CLIP, Z_CLIP)

    def _num_inv(self, Z):
        P = ndtr(np.clip(Z, -Z_CLIP, Z_CLIP))
        X = np.empty_like(Z)
        for j in range(Z.shape[1]):
            X[:, j] = islog(np.interp(P[:, j], self.probs, self.q[j]))
        X = np.clip(X, self.lo, self.hi)
        X[:, self.is_int] = np.round(X[:, self.is_int])
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full((len(df), self.dim), -1.0, dtype="float64")
        if self.num_cols:
            out[:, : len(self.num_cols)] = self._num_fwd(df[self.num_cols].to_numpy(dtype="float64"))
        for b in self.blocks:
            if b.kind == "cat":
                v = df[b.name].astype(str).to_numpy()
                idx = pd.Index(b.categories).get_indexer(v)
                idx = np.where(idx < 0, len(b.categories) - 1, idx)
                out[np.arange(len(df)), b.start + idx] = 1.0
            elif b.kind == "bit":
                out[:, b.start] = np.where(df[b.name].to_numpy() > 0, 1.0, -1.0)
        return out.astype("float32")

    def inverse_transform(self, Z: np.ndarray) -> pd.DataFrame:
        Z = np.asarray(Z, dtype="float64")
        data = {}
        if self.num_cols:
            Xn = self._num_inv(Z[:, : len(self.num_cols)])
            for j, c in enumerate(self.num_cols):
                data[c] = Xn[:, j]
        for b in self.blocks:
            if b.kind == "cat":
                data[b.name] = np.asarray(b.categories, dtype=object)[Z[:, b.start:b.end].argmax(1)]
            elif b.kind == "bit":
                data[b.name] = (Z[:, b.start] > 0).astype("int8")
        return pd.DataFrame(data)

    def snap(self, Z: np.ndarray) -> np.ndarray:
        """Snap an arbitrary encoded matrix (e.g. SMOTE-interpolated one-hots) back to valid codes."""
        return self.transform(self.inverse_transform(Z))

    def decode_like_generator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply to a raw generator output (original space) the same decoding that ``inverse_transform``
        applies to diffusion samples: integer columns rounded, numerical values clipped to the train
        range, unknown categories -> ``__other__``. Used so that ``validity_raw`` is comparable across
        generators (CTGAN/TVAE/TabSyn emit floats such as 3.2 packets, which is not a flow)."""
        out = df.copy()
        if self.num_cols:
            X = self._clean_arr(out[self.num_cols].to_numpy(dtype="float64"))
            X = np.clip(X, self.lo, self.hi)
            X[:, self.is_int] = np.round(X[:, self.is_int])
            out[self.num_cols] = X
        for b in self.blocks:
            if b.kind == "cat":
                v = out[b.name].astype(str)
                out[b.name] = v.where(v.isin(b.categories), "__other__").to_numpy(dtype=object)
            elif b.kind == "bit":
                out[b.name] = (pd.to_numeric(out[b.name], errors="coerce").fillna(0).to_numpy() > 0.5).astype("int8")
        return out

    # ------------------------------------------------------- torch (differentiable)
    def torch_buffers(self, device):
        import torch

        if getattr(self, "_tb_dev", None) != str(device):
            self._tb = dict(
                q=torch.tensor(self.q, dtype=torch.float32, device=device),
                probs=torch.tensor(self.probs, dtype=torch.float32, device=device),
                lo=torch.tensor(self.lo, dtype=torch.float32, device=device),
                hi=torch.tensor(self.hi, dtype=torch.float32, device=device),
            )
            self._tb_dev = str(device)
        return self._tb

    def decode_torch(self, Z, temp: float = 4.0):
        """Z (n, dim) -> (num (n, n_num) original values, soft dict {cat: probs, bit: prob})."""
        import torch

        tb = self.torch_buffers(Z.device)
        n_num = len(self.num_cols)
        soft = {}
        num = None
        if n_num:
            P = torch.special.ndtr(Z[:, :n_num].clamp(-Z_CLIP, Z_CLIP))
            num = _interp_rows(P, tb["probs"], tb["q"])
            num = torch.sign(num) * torch.expm1(num.abs().clamp(max=60.0))
            num = torch.minimum(torch.maximum(num, tb["lo"]), tb["hi"])
        for b in self.blocks:
            if b.kind == "cat":
                soft[b.name] = torch.softmax(temp * Z[:, b.start:b.end], dim=1)
            elif b.kind == "bit":
                soft[b.name] = torch.sigmoid(temp * Z[:, b.start])
        return num, soft


def _interp_rows(P, xp, Q):
    """Differentiable linear interpolation: for each column j, x = interp(P[:, j], xp, Q[j])."""
    import torch

    nq = xp.shape[0]
    idx = torch.searchsorted(xp, P.clamp(0, 1).contiguous()).clamp(1, nq - 1)  # (n, d)
    x0, x1 = xp[idx - 1], xp[idx]
    Qt = Q.t()  # (nq, d)
    y0 = torch.gather(Qt, 0, idx - 1)
    y1 = torch.gather(Qt, 0, idx)
    w = (P - x0) / (x1 - x0).clamp_min(1e-12)
    return y0 + w * (y1 - y0)
