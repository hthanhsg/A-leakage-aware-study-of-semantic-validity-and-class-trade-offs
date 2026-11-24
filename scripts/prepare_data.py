"""Phase A -- data preparation & audit.

python scripts/prepare_data.py --datasets nf_unsw_v3 nf_ton_v3 ...
Produces data/processed/<ds>/data.parquet and audit.json (columns, types, duplicates,
label conflicts, inf/NaN, constant columns, class distribution) -> used for the dataset table.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validaug.io import harmonize, read_raw, save_dataset  # noqa: E402
from validaug.schema import make_schema  # noqa: E402
from validaug.utils import LOG, setup_logging, write_env_snapshot  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    setup_logging()
    cfg = yaml.safe_load(open(a.config))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    write_env_snapshot(a.out)
    for name in a.datasets:
        dcfg = dict(cfg["datasets"][name], name=name)
        sch = make_schema(dcfg)
        raw = read_raw(dcfg["paths"])
        df, rep = harmonize(raw, sch, attempted=dcfg.get("attempted", "drop"), conflicts=dcfg.get("conflicts", "keep"),
                            label_map=dcfg.get("label_map"), drop_classes=dcfg.get("drop_classes"),
                            min_class_count=cfg["common"]["min_class_count"],
                            max_per_class=dcfg.get("max_per_class"), seed=cfg["common"]["seed"])
        save_dataset(df, rep, Path(a.out) / name)
        LOG.info("[%s] %d -> %d rows, %d features (%d categorical, %d bit), duplicates=%d, label conflicts=%d",
                 name, rep["rows_raw"], rep["rows_clean"], rep["n_features"], len(rep["categorical"]),
                 len(rep["bit_features"]), rep["exact_duplicates"], rep["label_conflicts"])
        LOG.info("[%s] classes: %s", name, rep["class_counts"])
        LOG.info("[%s] duplicates per class: %s", name, rep.get("exact_duplicates_by_class"))
        if rep.get("classes_dropped_config"):
            LOG.info("[%s] classes dropped by config: %s", name, rep["classes_dropped_config"])
        if not rep["has_time"] or not rep["has_host"]:
            LOG.warning("[%s] has_time=%s has_host=%s -> splits requiring these columns will be skipped",
                        name, rep["has_time"], rep["has_host"])
        if rep.get("inf_by_column"):
            LOG.info("[%s] columns with inf: %s", name, rep["inf_by_column"])
        if rep.get("label_conflicts"):
            LOG.info("[%s] label conflicts per class (%s): %s", name, rep["conflict_policy"], rep["label_conflicts_by_class"])
