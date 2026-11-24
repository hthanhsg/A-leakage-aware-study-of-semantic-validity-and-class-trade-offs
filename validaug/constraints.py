"""NetFlow/CICFlowMeter domain constraints: templates + thresholds calibrated on train.

Three outputs:
  * ``ConstraintSet.energy_torch``     : differentiable energy V(x) (guidance, paper Sec. 3).
  * ``ConstraintSet.repair``           : projection Pi_C in the original feature space (Sec. 6).
  * ``ConstraintSet.violation_report`` : violation rates (Validity axis).

Anti-leakage rule: thresholds (lo/hi of the bytes/packets ratio, the set of retained
constraints, learned invariants) are derived from the training partition only.
A template is kept as a *hard constraint* only if the real training data itself violates it
in at most ``tau_hard`` of the rows (default 0.5%).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import TCP_FLAG_BITS
from .utils import LOG

TOL = 1e-6


# ----------------------------------------------------------------- views
class NumpyView:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def num(self, c):
        return self.df[c].to_numpy(dtype="float64")

    def prob(self, c, value):
        return (self.df[c].astype(str).to_numpy() == str(value)).astype("float64")

    def bit(self, c):
        return self.df[c].to_numpy(dtype="float64")

    relu = staticmethod(lambda x: np.maximum(x, 0.0))
    abs = staticmethod(np.abs)


class TorchView:
    def __init__(self, num, soft, encoder):
        self._num, self.soft, self.enc = num, soft, encoder
        self._ix = {c: i for i, c in enumerate(encoder.num_cols)}

    def num(self, c):
        return self._num[:, self._ix[c]]

    def prob(self, c, value):
        import torch

        cats = self.enc.cats[c]
        if str(value) not in cats:
            return torch.zeros(self.soft[c].shape[0], device=self.soft[c].device)
        return self.soft[c][:, cats.index(str(value))]

    def bit(self, c):
        return self.soft[c]

    @staticmethod
    def relu(x):
        import torch

        return torch.relu(x)

    @staticmethod
    def abs(x):
        return x.abs()


# ------------------------------------------------------------ constraints
@dataclass
class Constraint:
    name: str
    hard: bool = True
    weight: float = 1.0

    def residual(self, v, scale):  # >= 0, and 0 when satisfied
        raise NotImplementedError

    def repair(self, df):  # fix df in place (original space)
        pass


@dataclass
class NonNeg(Constraint):
    col: str = ""

    def residual(self, v, s):
        return v.relu(-v.num(self.col)) / s[self.col]

    def repair(self, df):
        df[self.col] = df[self.col].clip(lower=0)


@dataclass
class Order(Constraint):
    a: str = ""
    b: str = ""
    mode: str = "mid"  # "mid": Euclidean projection; "lower_a": lower a to b (a is a sub-quantity)

    def residual(self, v, s):
        return v.relu(v.num(self.a) - v.num(self.b)) / s[self.b]

    def repair(self, df):
        a, b = df[self.a].to_numpy(float), df[self.b].to_numpy(float)
        bad = a > b + TOL
        if not bad.any():
            return
        if self.mode == "lower_a":
            a = np.where(bad, b, a)
        else:
            m = (a + b) / 2.0
            a, b = np.where(bad, m, a), np.where(bad, m, b)
        df[self.a], df[self.b] = a, b


@dataclass
class Ratio(Constraint):
    """lo*pkts <= bytes <= hi*pkts ; pkts = 0 => bytes = 0 (if train satisfies it)."""

    num_col: str = ""
    den_col: str = ""
    lo: float = 0.0
    hi: float = np.inf
    zero_coupled: bool = True

    def calibrate(self, df):
        n, d = df[self.num_col].to_numpy(float), df[self.den_col].to_numpy(float)
        pos = d > 0
        r = n[pos] / d[pos]
        self.lo, self.hi = float(np.min(r)), float(np.max(r))
        if (~pos).any():
            self.zero_coupled = bool(np.mean(n[~pos] > 0) < 0.005)

    def residual(self, v, s):
        n, d = v.num(self.num_col), v.num(self.den_col)
        r = v.relu(self.lo * d - n) + v.relu(n - self.hi * d)
        if not self.zero_coupled:  # only applies to flows with packets
            r = r * (d > 0.5)
        return r / s[self.num_col]

    def repair(self, df):
        n, d = df[self.num_col].to_numpy(float), df[self.den_col].to_numpy(float)
        lo, hi = np.ceil(self.lo * d - TOL), np.floor(self.hi * d + TOL)
        hi = np.maximum(hi, lo)
        fixed = np.clip(n, lo, hi)
        if not self.zero_coupled:
            fixed = np.where(d > 0.5, fixed, n)
        df[self.num_col] = fixed


@dataclass
class ZeroUnless(Constraint):
    """col must be 0 (or categorical value '0') unless cond_col is in cond_values."""

    col: str = ""
    kind: str = "num"  # num | bit | cat
    cond_col: str = ""
    cond_values: tuple = ("6",)

    def _p_cond(self, v):
        p = 0
        for val in self.cond_values:
            p = p + v.prob(self.cond_col, val)
        return p

    def residual(self, v, s):
        off = 1.0 - self._p_cond(v)
        if self.kind == "num":
            return v.abs(v.num(self.col)) / s[self.col] * off
        if self.kind == "bit":
            return v.bit(self.col) * off
        return (1.0 - v.prob(self.col, "0")) * off

    def repair(self, df):
        mask = ~df[self.cond_col].astype(str).isin([str(x) for x in self.cond_values])
        df.loc[mask, self.col] = "0" if self.kind == "cat" else 0


# -------------------------------------------------------------- templates
def _templates(family: str, cols: set, cat_cols: set, bit_cols: set, protocol_col: str | None):
    T = []

    def has(*c):
        return all(x in cols for x in c)

    if family == "netflow":
        for a, b in [("in_bytes", "in_pkts"), ("out_bytes", "out_pkts")]:
            if has(a, b):
                T.append(Ratio(f"ratio:{a}/{b}", num_col=a, den_col=b))
        mids = [("min_ip_pkt_len", "max_ip_pkt_len"), ("shortest_flow_pkt", "longest_flow_pkt"),
                ("min_ttl", "max_ttl")]
        for d in ("src_to_dst", "dst_to_src"):
            mids += [(f"{d}_iat_min", f"{d}_iat_avg"), (f"{d}_iat_avg", f"{d}_iat_max")]
        lows = [("retransmitted_in_bytes", "in_bytes"), ("retransmitted_in_pkts", "in_pkts"),
                ("retransmitted_out_bytes", "out_bytes"), ("retransmitted_out_pkts", "out_pkts"),
                ("duration_in", "flow_duration_milliseconds"), ("duration_out", "flow_duration_milliseconds")]
        tcp_only_num = ["tcp_win_max_in", "tcp_win_max_out", "retransmitted_in_bytes",
                        "retransmitted_in_pkts", "retransmitted_out_bytes", "retransmitted_out_pkts"]
        tcp_bits = [f"{p}_{b}" for p in ("tcp_flags", "client_tcp_flags", "server_tcp_flags") for b in TCP_FLAG_BITS]
        icmp_cats = ["icmp_type", "icmp_ipv4_type"]
    else:  # cicflowmeter
        T += [Ratio(f"ratio:{a}/{b}", num_col=a, den_col=b)
              for a, b in [("fwd_bytes", "fwd_pkts"), ("bwd_bytes", "bwd_pkts")] if has(a, b)]
        mids = []
        for p in ("fwd_pkt_len", "bwd_pkt_len", "pkt_len"):
            mids += [(f"{p}_min", f"{p}_mean"), (f"{p}_mean", f"{p}_max")]
        for p in ("flow_iat", "fwd_iat", "bwd_iat"):
            mids += [(f"{p}_min", f"{p}_mean"), (f"{p}_mean", f"{p}_max")]
        lows = [("fwd_iat_max", "fwd_iat_total"), ("bwd_iat_max", "bwd_iat_total"),
                ("fwd_iat_total", "flow_duration"), ("bwd_iat_total", "flow_duration")]
        tcp_only_num = [f"{b}_flag_count" for b in TCP_FLAG_BITS]
        tcp_bits, icmp_cats = [], []

    T += [Order(f"order:{a}<={b}", a=a, b=b, mode="mid") for a, b in mids if has(a, b)]
    T += [Order(f"sub:{a}<={b}", a=a, b=b, mode="lower_a") for a, b in lows if has(a, b)]
    if protocol_col and protocol_col in cat_cols:
        T += [ZeroUnless(f"tcp_only:{c}", col=c, kind="num", cond_col=protocol_col)
              for c in tcp_only_num if c in cols and c not in cat_cols]
        T += [ZeroUnless(f"tcp_only:{c}", col=c, kind="bit", cond_col=protocol_col)
              for c in tcp_bits if c in bit_cols]
        T += [ZeroUnless(f"icmp_only:{c}", col=c, kind="cat", cond_col=protocol_col, cond_values=("1", "58"))
              for c in icmp_cats if c in cat_cols]
    return T


class ConstraintSet:
    def __init__(self, family: str, tau_hard: float = 0.005, tau_learn: float = 0.001,
                 max_learned: int = 60, mine_learned: bool = True):
        self.family, self.tau_hard, self.tau_learn = family, tau_hard, tau_learn
        self.max_learned, self.mine_learned = max_learned, mine_learned

    def fit(self, df_train: pd.DataFrame, encoder, protocol_col: str | None):
        num, cat, bit = encoder.num_cols, encoder.cat_cols, encoder.bit_cols
        self.scale = {c: float(s) for c, s in zip(num, encoder.scale)}
        for c in cat + bit:
            self.scale[c] = 1.0
        cols = set(num) | set(cat) | set(bit)
        cand = [NonNeg(f"nonneg:{c}", col=c) for c, lo in zip(num, encoder.lo) if lo >= 0]
        cand += _templates(self.family, cols, set(cat), set(bit), protocol_col)
        view = NumpyView(df_train)
        self.hard, self.rejected = [], {}
        for c in cand:
            if isinstance(c, Ratio):
                c.calibrate(df_train)
            if isinstance(c, ZeroUnless) and c.kind == "cat" and "0" not in encoder.cats.get(c.col, []):
                # the repair value "0" is not an encodable category -> the constraint could never be
                # satisfied after re-encoding; reject it explicitly (reported like a train violation)
                self.rejected[c.name] = -1.0  # -1 = "repair value not encodable", not a violation rate
                continue
            rate = float(np.mean(c.residual(view, self.scale) > TOL))
            if rate <= self.tau_hard:
                self.hard.append(c)
            else:
                self.rejected[c.name] = rate
        self.soft = self._mine(df_train, num) if self.mine_learned else []
        LOG.info("Constraints: %d hard, %d rejected (train violation > %.3f), %d learned invariants",
                 len(self.hard), len(self.rejected), self.tau_hard, len(self.soft))
        if self.rejected:
            LOG.info("Rejected templates (train violation rate): %s",
                     {k: round(v, 4) for k, v in self.rejected.items()})
        return self

    def _mine(self, df, num, max_rows=50_000, seed=0):
        """Mine invariants a <= b (Daikon-style) on train; used as soft constraints only."""
        rng = np.random.default_rng(seed)
        sub = df if len(df) <= max_rows else df.iloc[rng.choice(len(df), max_rows, replace=False)]
        X = sub[num].to_numpy(float)
        X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
        covered = {(c.a, c.b) for c in self.hard if isinstance(c, Order)}
        ok = [j for j in range(X.shape[1]) if np.ptp(X[:, j]) > 0]
        found = []
        for i in ok:
            for j in ok:
                if i == j or (num[i], num[j]) in covered:
                    continue
                d = X[:, i] - X[:, j]
                viol = np.mean(d > TOL * (1 + np.abs(X[:, j])))
                if viol <= self.tau_learn and np.mean(d < -TOL) >= 0.05:
                    tight = float(np.mean(np.abs(d)) / (np.std(X[:, j]) + 1e-9))
                    found.append((viol, tight, num[i], num[j]))
        found.sort()
        return [Order(f"learned:{a}<={b}", a=a, b=b, hard=False, weight=0.5)
                for _, _, a, b in found[: self.max_learned]]

    # ---------------------------------------------------------- export
    def describe(self) -> dict:
        """Serializable description of the fitted constraint set (for the paper appendix)."""
        def one(c):
            d = {"name": c.name, "type": type(c).__name__, "hard": c.hard, "weight": c.weight}
            for k in ("col", "a", "b", "mode", "num_col", "den_col", "lo", "hi", "zero_coupled", "kind",
                      "cond_col", "cond_values"):
                if hasattr(c, k):
                    v = getattr(c, k)
                    d[k] = list(v) if isinstance(v, tuple) else v
            return d
        return {"family": self.family, "tau_hard": self.tau_hard, "tau_learn": self.tau_learn,
                "hard": [one(c) for c in self.hard], "soft": [one(c) for c in self.soft],
                "rejected_train_violation": self.rejected}

    # ---------------------------------------------------------- energy
    def energy_torch(self, num, soft, encoder):
        import torch

        v = TorchView(num, soft, encoder)
        E = torch.zeros(num.shape[0] if num is not None else next(iter(soft.values())).shape[0],
                        device=(num if num is not None else next(iter(soft.values()))).device)
        for c in self.hard + self.soft:
            E = E + c.weight * c.residual(v, self.scale) ** 2
        return E

    # ---------------------------------------------------------- validity
    def violation_report(self, df: pd.DataFrame, include_soft: bool = False) -> dict:
        v = NumpyView(df)
        cs = self.hard + (self.soft if include_soft else [])
        if not cs or len(df) == 0:
            return {"any": 0.0, "per_constraint": {}}
        M = np.stack([c.residual(v, self.scale) > TOL for c in cs], 1)
        return {"any": float(M.any(1).mean()),
                "mean_per_constraint": float(M.mean()),
                "per_constraint": {c.name: float(M[:, k].mean()) for k, c in enumerate(cs)}}

    # ---------------------------------------------------------- projection
    def repair(self, df: pd.DataFrame, encoder, n_iter: int = 6, include_soft: bool = False) -> pd.DataFrame:
        """Pi_C: cyclic projections (POCS) over the hard constraints, then integer rounding.

        Order: ZeroUnless (discrete) -> NonNeg -> Order -> Ratio (bytes given packets).
        Count variables (packets) are kept fixed and bytes are clipped -> corresponds to a metric
        M with a very large weight on count variables (see paper).
        """
        df = df.copy()
        cs = self.hard + (self.soft if include_soft else [])
        order = [ZeroUnless, NonNeg, Order, Ratio]
        cs = sorted(cs, key=lambda c: order.index(type(c)))
        int_cols = [c for c, f in zip(encoder.num_cols, encoder.is_int) if f]
        chains = _chains([c for c in cs if isinstance(c, Order) and c.mode == "mid"])
        covered = {(ch[i], ch[i + 1]) for ch in chains for i in range(len(ch) - 1)}
        for _ in range(n_iter):
            for c in cs:
                if isinstance(c, Order) and (c.a, c.b) in covered:
                    continue
                c.repair(df)
            for ch in chains:  # min <= mean <= max: row-wise sort (keeps integrality)
                df[ch] = np.sort(df[ch].to_numpy(float), axis=1)
            df[int_cols] = np.round(df[int_cols].astype(float))
            if self.violation_report(df)["any"] == 0.0:
                break
        return df


def _chains(orders):
    """Merge pairs a<=b into chains a<=b<=c... (path graph)."""
    nxt = {o.a: o.b for o in orders}
    heads = set(nxt) - set(nxt.values())
    out = []
    for h in heads:
        ch = [h]
        while ch[-1] in nxt and nxt[ch[-1]] not in ch:
            ch.append(nxt[ch[-1]])
        out.append(ch)
    return out
