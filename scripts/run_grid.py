"""Run an experiment grid.  python scripts/run_grid.py --exp configs/phaseD_main.yaml"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validaug.runner import Runner  # noqa: E402
from validaug.utils import setup_logging  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--datasets-config", default="configs/datasets.yaml")
    ap.add_argument("--only-datasets", nargs="*", help="override the dataset list (to split work across GPUs)")
    ap.add_argument("--only-methods", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--set", nargs="*", default=[], help="override nested keys, e.g. aug.validaug.lambda_c=0.5")
    ap.add_argument("--out", help="override out_dir")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.exp))
    for kv in a.set:
        k, v = kv.split("=", 1)
        node = cfg
        *path, last = k.split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(v)
    if a.out:
        cfg["out_dir"] = a.out
    dcfg = yaml.safe_load(open(a.datasets_config))
    cfg.setdefault("label_groups", dcfg.get("label_groups", {}))
    if a.only_datasets is not None:
        cfg["datasets"] = a.only_datasets
        cfg["cross_dataset"] = [p for p in cfg.get("cross_dataset", []) if p["train"] in a.only_datasets]
    if a.only_methods:
        cfg["methods"] = a.only_methods
    if a.seeds:
        cfg["seeds"] = a.seeds
    setup_logging(cfg.get("log_level", "INFO"), str(Path(cfg["out_dir"]) / "run.log"))
    Runner(cfg, dcfg["datasets"]).run()
