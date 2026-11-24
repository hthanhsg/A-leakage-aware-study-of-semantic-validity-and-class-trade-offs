"""Build a reproducibility package (e.g. for Zenodo / reviewers).

python scripts/make_artifact.py --out artifact [--with-synthetic] [--with-models] [--runs runs/phaseB_protocol ...]

Included by default
  * code: validaug/, scripts/, configs/, tests/, README.md, requirements.txt, run_all.sh
  * data audits: data/processed/*/audit.json (+ environment snapshot of the preparation step)
  * per run directory: results.jsonl, config_used.json, environment_*.json, run.log (gzip),
    splits/*.json + *.npz (exact train/val/test indices), constraints/*.json, analysis/**
  * tuned guidance parameters: runs/tune_*.yaml
Optional
  * --with-synthetic : synthetic flow samples (runs/*/synthetic/**.parquet)
  * --with-models    : cached diffusion models (runs/cache/**/diffusion_*.pt) -- large
Never included: raw or processed datasets (redistribution is governed by the dataset licences;
cite the original sources instead).

A MANIFEST.json with the SHA-256 of every file is written, and the package is archived as .tar.gz.
"""
import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ["validaug", "scripts", "configs", "tests", "README.md", "requirements.txt", "run_all.sh", ".gitignore"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def copy(src: Path, dst: Path, gz: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if gz:
        with open(src, "rb") as fi, gzip.open(str(dst) + ".gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
    else:
        shutil.copy2(src, dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifact")
    ap.add_argument("--runs", nargs="*", help="run directories (default: every directory in runs/)")
    ap.add_argument("--with-synthetic", action="store_true")
    ap.add_argument("--with-models", action="store_true")
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%d")
    pkg = Path(a.out) / f"validaug_artifact_{stamp}"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)

    for item in CODE:
        src = ROOT / item
        if src.is_dir():
            shutil.copytree(src, pkg / item, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif src.exists():
            copy(src, pkg / item)

    proc = ROOT / "data" / "processed"
    for f in list(proc.glob("*/audit.json")) + list(proc.glob("environment_*.json")):
        copy(f, pkg / "data_audit" / f.relative_to(proc))

    runs = [Path(r) for r in a.runs] if a.runs else sorted(p for p in (ROOT / "runs").glob("*") if p.is_dir())
    for r in runs:
        r = r if r.is_absolute() else ROOT / r
        dst = pkg / "runs" / r.name
        for name in ["results.jsonl", "config_used.json"]:
            if (r / name).exists():
                copy(r / name, dst / name)
        for f in r.glob("environment_*.json"):
            copy(f, dst / f.name)
        if (r / "run.log").exists():
            copy(r / "run.log", dst / "run.log", gz=True)
        for sub, pats in [("splits", ["*.json", "*.npz"]), ("constraints", ["*.json"]), ("analysis", ["**/*"])]:
            for pat in pats:
                for f in (r / sub).glob(pat):
                    if f.is_file():
                        copy(f, dst / sub / f.relative_to(r / sub))
        if a.with_synthetic:
            for f in (r / "synthetic").glob("**/*.parquet"):
                copy(f, dst / "synthetic" / f.relative_to(r / "synthetic"))
        if a.with_models:
            for f in (r / "cache").glob("**/diffusion*.pt"):
                copy(f, dst / "models" / f.relative_to(r / "cache"))
    if a.with_models:  # shared cache (cache_root: runs/cache)
        for f in (ROOT / "runs" / "cache").glob("**/diffusion*.pt"):
            copy(f, pkg / "runs" / "cache" / f.relative_to(ROOT / "runs" / "cache"))
    for f in (ROOT / "runs").glob("tune_*.yaml"):
        copy(f, pkg / "runs" / f.name)

    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {}}
    total = 0
    for f in sorted(pkg.rglob("*")):
        if f.is_file():
            manifest["files"][str(f.relative_to(pkg))] = {"sha256": sha256(f), "bytes": f.stat().st_size}
            total += f.stat().st_size
    manifest["total_bytes"] = total
    (pkg / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))

    tar = Path(a.out) / f"{pkg.name}.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(pkg, arcname=pkg.name)
    print(f"Package: {tar}  ({tar.stat().st_size / 2**20:.1f} MiB, {len(manifest['files'])} files)")
    print(f"SHA-256: {sha256(tar)}")
