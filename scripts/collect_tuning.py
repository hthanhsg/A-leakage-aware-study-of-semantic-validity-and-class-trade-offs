"""Collect the phase-C results (runs/tune_<dataset>.yaml) into an `aug_per_dataset:` block to paste into
configs/phaseD_main.yaml, phaseE_ablation.yaml and phaseF_generalization.yaml.

Only the five guidance parameters are taken (lambda_c, lambda_b, delta, t1, t2): phase C runs with
`pool_cap`/`oversample` overrides that must NOT leak into phase D.

python scripts/collect_tuning.py --tune-dir runs            # prints the YAML block
python scripts/collect_tuning.py --tune-dir runs --apply configs/phaseD_main.yaml configs/phaseE_ablation.yaml
"""
import argparse
from pathlib import Path

import yaml

KEYS = ("lambda_c", "lambda_b", "delta", "t1", "t2")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-dir", default="runs")
    ap.add_argument("--apply", nargs="*", default=[], help="config files to update in place (aug_per_dataset)")
    a = ap.parse_args()
    block = {}
    for p in sorted(Path(a.tune_dir).glob("tune_*.yaml")):
        ds = p.stem[len("tune_"):]
        vc = yaml.safe_load(p.read_text())["validaug"]
        block[ds] = {"validaug": {k: float(vc[k]) for k in KEYS if k in vc}}
        head = p.read_text().splitlines()[0]
        print(f"# {ds}: {head.lstrip('# ')}")
    text = yaml.safe_dump({"aug_per_dataset": block}, sort_keys=False)
    print(text)
    for cfg_path in a.apply:
        cfg = yaml.safe_load(open(cfg_path))
        cfg["aug_per_dataset"] = {**cfg.get("aug_per_dataset", {}), **block}
        raw = Path(cfg_path).read_text()
        if "aug_per_dataset:" in raw:  # replace the existing block (it is always at the end of the file)
            raw = raw[: raw.index("aug_per_dataset:")]
        Path(cfg_path).write_text(raw.rstrip("\n") + "\n\n# ---- phase C results (scripts/collect_tuning.py) ----\n"
                                  + yaml.safe_dump({"aug_per_dataset": cfg["aug_per_dataset"]}, sort_keys=False))
        print(f"# updated {cfg_path}")
