"""Splitting protocols.

* ``random_naive``        : stratified random split WITHOUT duplicate removal or conflict resolution
                            (reproduces common literature practice; upper bound of inflation).
* ``random``              : stratified random split (deduplicated; isolates non-duplicate leakage).
* ``temporal_stratified`` : within each class, sort by time and cut 60/20/20 (default).
* ``temporal_global``     : global temporal cut (classes may vanish from a partition -> reported).
* ``group``               : within each class, split by source-host groups (no host appears in two
                            partitions of the same class); classes with < 3 hosts fall back to temporal_stratified.

After splitting, every protocol (unless disabled) removes duplicate leakage: val/test rows whose
(features, label) exactly match a training row are dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .io import feature_columns
from .utils import LOG

FRACS = (0.6, 0.2, 0.2)


def _cut(idx_sorted, fr=FRACS):
    n = len(idx_sorted)
    a, b = int(round(n * fr[0])), int(round(n * (fr[0] + fr[1])))
    return idx_sorted[:a], idx_sorted[a:b], idx_sorted[b:]


def _random(df, seed):
    idx = np.arange(len(df))
    y = df["__y"].to_numpy()
    tr, rest = train_test_split(idx, train_size=FRACS[0], stratify=y, random_state=seed)
    va, te = train_test_split(rest, train_size=FRACS[1] / (FRACS[1] + FRACS[2]), stratify=y[rest], random_state=seed)
    return tr, va, te, {}


def _temporal_stratified(df, seed, classes=None):
    out = [[], [], []]
    t = df["__time"].to_numpy()
    if np.all(np.isnan(t)):
        raise ValueError("No timestamp column available")
    rng = np.random.default_rng(seed)
    for cls, idx in df.groupby("__y").indices.items():
        if classes is not None and cls not in classes:
            continue
        tie = rng.random(len(idx))  # seeded random tie-breaking
        order = idx[np.lexsort((tie, np.nan_to_num(t[idx], nan=np.inf)))]
        for k, part in enumerate(_cut(order)):
            out[k].append(part)
    return [np.concatenate(o) if o else np.array([], int) for o in out]


def _temporal_global(df, seed):
    t = np.nan_to_num(df["__time"].to_numpy(), nan=np.inf)
    order = np.lexsort((np.random.default_rng(seed).random(len(df)), t))
    tr, va, te = _cut(order)
    y = df["__y"].to_numpy()
    missing = {s: sorted(set(y) - set(y[i])) for s, i in zip(("train", "val", "test"), (tr, va, te))}
    return tr, va, te, {"classes_missing": missing}


def _group(df, seed):
    rng = np.random.default_rng(seed)
    out, fallback = [[], [], []], []
    for cls, idx in df.groupby("__y").indices.items():
        hosts = df["__host"].to_numpy()[idx]
        uh, cnt = np.unique(hosts, return_counts=True)
        if len(uh) < 3:
            fallback.append(cls)
            continue
        perm = rng.permutation(len(uh))
        uh, cnt = uh[perm], cnt[perm]
        # greedy: assign the largest hosts first to the partition furthest below its target
        order = np.argsort(-cnt, kind="stable")
        target = np.array(FRACS) * len(idx)
        fill = np.zeros(3)
        assign = {}
        for k in range(3):  # make sure each partition gets at least one host
            assign[uh[order[k]]] = k; fill[k] += cnt[order[k]]
        for o in order[3:]:
            k = int(np.argmax(target - fill))
            assign[uh[o]] = k; fill[k] += cnt[o]
        part = np.array([assign[h] for h in hosts])
        for k in range(3):
            out[k].append(idx[part == k])
    if fallback:
        LOG.warning("Group split: classes %s have < 3 hosts -> falling back to temporal_stratified", fallback)
        extra = _temporal_stratified(df, seed, classes=set(fallback))
        for k in range(3):
            out[k].append(extra[k])
    tr, va, te = [np.concatenate(o) for o in out]
    return tr, va, te, {"group_fallback_classes": fallback}


def make_split(df: pd.DataFrame, kind: str, seed: int, dedup: bool = True,
               train_conflicts: str = "majority") -> dict:
    if kind == "random_naive":  # literature practice: no conflict resolution, no deduplication
        tr, va, te, info = _random(df, seed)
        h = _row_hash(df, feature_columns(df) + ["__y"])
        info["naive_overlap"] = {"test_rows_identical_to_a_train_row": int(np.isin(h[te], h[tr]).sum()),
                                 "test_rows": int(len(te))}
        info["sizes"] = [len(tr), len(va), len(te)]
        return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te), "info": info}
    if kind == "random":
        tr, va, te, info = _random(df, seed)
    elif kind == "temporal_stratified":
        tr, va, te = _temporal_stratified(df, seed); info = {}
    elif kind == "temporal_global":
        tr, va, te, info = _temporal_global(df, seed)
    elif kind == "group":
        tr, va, te, info = _group(df, seed)
    else:
        raise ValueError(kind)
    if train_conflicts != "keep":
        tr, cinfo = resolve_train_conflicts(df, tr, train_conflicts)
        info.update(cinfo)
    info["sizes_before_dedup"] = [len(tr), len(va), len(te)]
    if dedup:
        tr, va, te, dinfo = dedup_leakage(df, tr, va, te)
        info.update(dinfo)
    info["sizes"] = [len(tr), len(va), len(te)]
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te), "info": info}


def majority_keep(key: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of rows to keep: in every group of identical feature vectors with more than one label, keep
    the rows carrying the majority label; ties -> drop the whole group. Non-conflicting groups are kept."""
    d = pd.DataFrame({"k": key, "y": y, "i": np.arange(len(y))})
    nlab = d.groupby("k")["y"].transform("nunique")
    ok = d[nlab == 1]["i"].to_numpy()
    c = d[nlab > 1]
    if len(c) == 0:
        return np.sort(ok)
    cnt = c.groupby(["k", "y"]).size().rename("n").reset_index()
    top = cnt.sort_values("n", ascending=False).groupby("k").head(2)
    best = top.groupby("k").agg(n1=("n", "max"), nmin=("n", "min"), cnt=("n", "size"))
    win = cnt.merge(best, on="k")
    win = win[(win.n == win.n1) & ~((win.cnt == 2) & (win.nmin == win.n1))]  # drop tied groups
    keep_c = c.merge(win[["k", "y"]], on=["k", "y"])["i"].to_numpy()
    return np.sort(np.concatenate([ok, keep_c]))


def resolve_train_conflicts(df, tr, policy="majority"):
    cols = feature_columns(df)
    key = _row_hash(df.iloc[tr], cols)
    y = df["__y"].to_numpy(object)[tr]
    if policy == "majority":
        keep = majority_keep(key, y)
    elif policy == "drop":
        nlab = pd.Series(y).groupby(key).transform("nunique").to_numpy()
        keep = np.where(nlab == 1)[0]
    else:
        raise ValueError(policy)
    removed = pd.Series(y[np.setdiff1d(np.arange(len(tr)), keep)]).value_counts().to_dict()
    return tr[keep], {"train_conflicts_policy": policy, "train_conflict_removed": {k: int(v) for k, v in removed.items()}}


def _row_hash(df, cols):
    return pd.util.hash_pandas_object(df[cols], index=False).to_numpy()


def dedup_leakage(df, tr, va, te):
    cols = feature_columns(df) + ["__y"]
    h = _row_hash(df, cols)
    # deduplicate within train (keep one copy) to avoid implicit sample weights
    _, first = np.unique(h[tr], return_index=True)
    tr2 = tr[np.sort(first)]
    seen = set(h[tr2])
    va2 = va[~np.isin(h[va], list(seen))]
    seen_tv = seen | set(h[va2])
    te2 = te[~np.isin(h[te], list(seen_tv))]
    info = {"dedup_removed": {"train_internal": int(len(tr) - len(tr2)),
                              "val_overlap_train": int(len(va) - len(va2)),
                              "test_overlap_trainval": int(len(te) - len(te2))}}
    return tr2, va2, te2, info


# ------------------------------------------------------------ scenarios
def apply_scenario(train_idx, val_idx, y: np.ndarray, scenario: dict, seed: int, benign="Benign"):
    """Scarcity / held-out-attack scenarios (only train/val are modified).

    scenario: {"type": "full"} | {"type": "rare", "frac": 0.05, "classes": "auto"|[...]}
              | {"type": "holdout", "cls": "Backdoor"}
    """
    st = scenario.get("type", "full")
    rng = np.random.default_rng(seed)
    if st == "full":
        return train_idx, val_idx, {}
    if st == "rare":
        frac = scenario["frac"]
        classes = scenario.get("classes", "auto")
        if classes == "auto":  # every attack class except the largest attack class
            cnt = pd.Series(y[train_idx]).value_counts()
            att = [c for c in cnt.index if c != benign]
            classes = att[1:] if len(att) > 1 else att
        keep = []
        for c in np.unique(y[train_idx]):
            ix = train_idx[y[train_idx] == c]
            if c in classes:
                k = max(int(scenario.get("min_keep", 10)), int(round(len(ix) * frac)))
                ix = rng.choice(ix, min(k, len(ix)), replace=False)
            keep.append(ix)
        return np.sort(np.concatenate(keep)), val_idx, {"rare_classes": list(classes)}
    if st == "holdout":
        c = resolve_class(scenario.get("cls", "auto"), y[train_idx], benign)
        if c is None:
            return None, None, {"skip": f"holdout class '{scenario.get('cls')}' not found"}
        return train_idx[y[train_idx] != c], val_idx[y[val_idx] != c], {"holdout": c}
    raise ValueError(st)


def resolve_class(spec, y_train, benign="Benign", min_count=50):
    """Resolve the real class name from a flexible spec, so configs need NOT be edited per dataset.

    spec = "auto"              -> rarest attack class with >= min_count training samples
    spec = "backdoor"          -> case-insensitive match (exact first, then substring)
    spec = ["backdoor", "bot"] -> try each candidate in turn
    """
    cnt = pd.Series(y_train).value_counts()
    att = cnt[cnt.index != benign]
    if isinstance(spec, str) and spec.lower() == "auto":
        ok = att[att >= min_count]
        return ok.idxmin() if len(ok) else None
    for cand in ([spec] if isinstance(spec, str) else list(spec)):
        cl = str(cand).lower()
        exact = [c for c in att.index if str(c).lower() == cl]
        if exact:
            return exact[0]
        part = [c for c in att.index if cl in str(c).lower()]
        if part:
            return max(part, key=lambda c: att[c])
    return None


def map_label_group(labels, groups: dict) -> np.ndarray:
    kw = sorted(((k.lower(), g) for g, ks in groups.items() for k in ks), key=lambda x: -len(x[0]))
    out = []
    for lab in labels:
        l = str(lab).lower()
        out.append(next((g for k, g in kw if k in l), "Other"))
    return np.array(out, dtype=object)
