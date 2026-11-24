"""Read, harmonize and clean raw NIDS data.

The output of ``harmonize`` is a DataFrame with:
  * feature columns (normalized names, bitmasks split into bits, destination port bucketed)
  * ``__y``    : multi-class label as string (benign variants mapped to "Benign")
  * ``__time`` : timestamp (ms) if available -> used ONLY for splitting, never a feature
  * ``__host`` : source-host identifier if available -> used ONLY for group splitting

Every step here is *data-independent* (nothing is fitted), so it is safe to run before the
train/val/test split. Fitted steps (imputation, scaling, constraint thresholds) live in
``encoder.py`` and ``constraints.py`` and are fitted on the training partition only.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import CIC_ALIASES, TCP_FLAG_BITS, DatasetSchema, norm_col
from .utils import LOG

META_COLS = ["__y", "__time", "__host"]


def read_raw(paths: list[str] | str, usecols=None, nrows=None) -> pd.DataFrame:
    if isinstance(paths, str):
        paths = [paths]
    files = sorted({f for p in paths for f in glob.glob(p)})
    if not files:
        raise FileNotFoundError(f"No file matches {paths}")
    frames = []
    for f in files:
        LOG.info("Reading %s", f)
        if f.endswith(".parquet"):
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f, low_memory=False, nrows=nrows, encoding_errors="replace")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _normalize_columns(df: pd.DataFrame, sch: DatasetSchema) -> pd.DataFrame:
    cols = [norm_col(c) for c in df.columns]
    if sch.family == "cicflowmeter":
        cols = [CIC_ALIASES.get(c, c) for c in cols]
    df.columns = cols
    # CIC-IDS2017 has a duplicated column (e.g. fwd_header_length_1) -> drop the copy
    dup = df.columns.duplicated()
    if dup.any():
        LOG.warning("Dropping %d duplicated columns: %s", dup.sum(), list(df.columns[dup]))
        df = df.loc[:, ~dup]
    if "fwd_header_length_1" in df.columns:
        df = df.drop(columns=["fwd_header_length_1"])
    return df


def _clean_labels(df: pd.DataFrame, sch: DatasetSchema, attempted: str) -> pd.DataFrame:
    y = df[sch.label_col].astype(str).str.strip()
    y = y.str.replace("�", "-", regex=False).str.replace(r"\s+", " ", regex=True)
    is_benign = y.str.lower().isin([b.lower() for b in sch.benign_values])
    y = y.where(~is_benign, "Benign")
    # "Attempted" labels of the corrected CIC-IDS2017 release (Engelen et al.)
    att = y.str.contains("attempted", case=False, regex=False)
    if att.any():
        LOG.info("%d flows labelled 'Attempted' -> policy: %s", att.sum(), attempted)
        if attempted == "benign":
            y = y.where(~att, "Benign")
        elif attempted == "drop":
            df = df.loc[~att].copy()
            y = y.loc[~att]
    df["__y"] = y.values
    return df


def _parse_time(df: pd.DataFrame, sch: DatasetSchema) -> pd.Series | None:
    if not sch.time_col or sch.time_col not in df.columns:
        return None
    t = df[sch.time_col]
    if pd.api.types.is_numeric_dtype(t):
        return t.astype("float64")
    # ISO "2017-07-03 08:55:58" (Engelen release) -> do NOT use dayfirst (dateutil would swap month/day);
    # "3/7/2017 8:55" (original UNB release) -> day/month/year.
    sample = t.dropna().astype(str).head(1000)
    iso = sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9 if len(sample) else False
    ts = pd.to_datetime(t, errors="coerce", dayfirst=not iso, format="ISO8601" if iso else "mixed")
    return ((ts - pd.Timestamp("1970-01-01")) / pd.Timedelta(milliseconds=1)).astype("float64")


def _split_bitmask(df: pd.DataFrame, col: str) -> pd.DataFrame:
    v = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64").to_numpy()
    for i, b in enumerate(TCP_FLAG_BITS):
        df[f"{col}_{b}"] = ((v >> i) & 1).astype("int8")
    return df.drop(columns=[col])


def _port_bucket(p: pd.Series, top_ports: list[int]) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce").fillna(-1).astype("int64")
    out = np.where(p < 0, "na", np.where(p < 1024, "wk_other", np.where(p < 49152, "reg_other", "dyn")))
    out = pd.Series(out, index=p.index, dtype="object")
    mask = p.isin(top_ports)
    out[mask] = p[mask].astype(str)
    return out


def harmonize(df: pd.DataFrame, sch: DatasetSchema, *, attempted: str = "drop", conflicts: str = "keep",
              label_map: dict | None = None, drop_classes: list | None = None,
              top_ports: list[int] | None = None, min_class_count: int = 50,
              max_per_class: int | None = None, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    report: dict = {"dataset": sch.name, "rows_raw": int(len(df))}
    df = _normalize_columns(df, sch)
    if sch.label_col not in df.columns:
        raise KeyError(f"[{sch.name}] Label column '{sch.label_col}' not found. Columns: {list(df.columns)[:20]}...")
    df = _clean_labels(df, sch, attempted)
    # Label mapping / class removal from config (e.g. classes documented as mislabelled)
    if label_map:
        df["__y"] = df["__y"].replace(label_map)
    if drop_classes:
        m = df["__y"].isin(drop_classes)
        report["classes_dropped_config"] = df.loc[m, "__y"].value_counts().to_dict()
        df = df.loc[~m].copy()
    df["__time"] = _parse_time(df, sch) if sch.time_col in df.columns else np.nan
    df["__host"] = df[sch.host_col].astype(str) if sch.host_col and sch.host_col in df.columns else "na"

    drop = set(sch.id_cols) | set(sch.extra_drop) | {sch.label_col}
    if sch.binary_label_col:
        drop.add(sch.binary_label_col)
    for pc, mode in sch.port_cols.items():
        if pc in df.columns and mode == "drop":
            drop.add(pc)
    df = df.drop(columns=[c for c in drop if c in df.columns])

    # Destination port -> categorical (fixed list of common ports + 3 IANA buckets); the port list is
    # fixed in code (not learned from data) to avoid leakage.
    top_ports = top_ports or [20, 21, 22, 23, 25, 53, 67, 68, 69, 80, 110, 123, 137, 138, 139,
                              143, 161, 389, 443, 445, 1883, 1900, 3306, 3389, 5353, 8080, 8443]
    for pc, mode in sch.port_cols.items():
        if pc in df.columns and mode == "categorical":
            df[pc] = _port_bucket(df[pc], top_ports)
    for bc in sch.bitmask_cols:
        if bc in df.columns:
            df = _split_bitmask(df, bc)
    if "l7_proto" in df.columns:  # nDPI "7.178" -> master protocol 7
        df["l7_proto"] = pd.to_numeric(df["l7_proto"], errors="coerce").fillna(0).astype("int64")

    cat_cols = [c for c in sch.categorical_cols if c in df.columns] + \
               [c for c in sch.port_cols if c in df.columns and sch.port_cols[c] == "categorical"]
    feat_cols = [c for c in df.columns if c not in META_COLS]
    num_cols = [c for c in feat_cols if c not in cat_cols and not _is_bit(c, sch)]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    arr = df[num_cols].to_numpy(dtype="float64")
    # +-inf (typically a division by a zero duration, e.g. throughput) is KEPT here; the encoder
    # replaces it with the finite TRAIN max/min (semantically "very large"), not with the median.
    inf_cols = {c: int(np.isinf(arr[:, j]).sum()) for j, c in enumerate(num_cols) if np.isinf(arr[:, j]).any()}
    report.update(n_inf=int(sum(inf_cols.values())), inf_by_column=inf_cols, n_nan=int(np.isnan(arr).sum()),
                  nan_by_column={c: int(np.isnan(arr[:, j]).sum()) for j, c in enumerate(num_cols) if np.isnan(arr[:, j]).any()})
    for c in cat_cols:  # "6.0" and "6" must be the same categorical value
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() > 0.99:
            df[c] = num.fillna(-1).round().astype("int64").astype(str)
        else:
            df[c] = df[c].astype(str)

    # Constant columns (over the whole dataset, label-independent) -> drop
    const = [c for c in feat_cols if df[c].nunique(dropna=False) <= 1]
    if const:
        df = df.drop(columns=const)
    report["constant_dropped"] = const

    # Classes with too few samples -> drop (reported)
    vc = df["__y"].value_counts()
    small = vc[vc < min_class_count].index.tolist()
    if small:
        df = df[~df["__y"].isin(small)]
    report["classes_dropped_small"] = {k: int(vc[k]) for k in small}

    # Per-class cap (very large datasets such as NF-BoT-IoT-v3). Seeded random sampling;
    # __time is preserved, so temporal splitting is still possible.
    if max_per_class:
        rng = np.random.default_rng(seed)
        keep = []
        for _, idx in df.groupby("__y").indices.items():
            keep.append(idx if len(idx) <= max_per_class else rng.choice(idx, max_per_class, replace=False))
        df = df.iloc[np.sort(np.concatenate(keep))]
    df = df.reset_index(drop=True)

    feats = [c for c in df.columns if c not in META_COLS]
    # Label conflicts: identical feature vectors with different labels -> label noise.
    nlab = df.groupby(feats, dropna=False, sort=False)["__y"].transform("nunique")
    conflict = (nlab > 1).to_numpy()
    report["label_conflicts_by_class"] = df.loc[conflict, "__y"].value_counts().to_dict()
    # Default "keep": report only. Conflicts are resolved AFTER splitting, on train only
    # (splits.resolve_train_conflicts), so val/test labels are never used for cleaning.
    if conflicts in ("drop", "majority") and conflict.any():
        key = pd.util.hash_pandas_object(df[feats], index=False).to_numpy()
        if conflicts == "drop":
            df = df.loc[~conflict].reset_index(drop=True)
        else:
            from .splits import majority_keep
            df = df.iloc[majority_keep(key, df["__y"].to_numpy(object))].reset_index(drop=True)
        report["class_counts_after_conflict_policy"] = {k: int(v) for k, v in df["__y"].value_counts().items()}
    dup_mask = df.duplicated(subset=feats + ["__y"])
    report.update(conflict_policy=conflicts)
    report.update(
        rows_clean=int(len(df)),
        n_features=len(feats),
        features=feats,
        categorical=[c for c in cat_cols if c in feats],
        bit_features=[c for c in feats if _is_bit(c, sch)],
        class_counts={k: int(v) for k, v in df["__y"].value_counts().items()},
        exact_duplicates=int(dup_mask.sum()),
        label_conflicts=int(conflict.sum()),
        exact_duplicates_by_class=df.loc[dup_mask, "__y"].value_counts().to_dict(),
        has_time=bool(df["__time"].notna().any()),
        has_host=bool((df["__host"] != "na").any()),
    )
    return df, report


def _is_bit(c: str, sch: DatasetSchema) -> bool:
    return any(c.startswith(f"{b}_") and c.rsplit("_", 1)[-1] in TCP_FLAG_BITS for b in sch.bitmask_cols)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def save_dataset(df: pd.DataFrame, report: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "data.parquet", index=False)
    from .utils import dump_json

    dump_json(report, out / "audit.json")
