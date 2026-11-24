"""Common utilities: seeding, logging, timers, JSON I/O, environment snapshot."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

LOG = logging.getLogger("validaug")


def setup_logging(level: str = "INFO", logfile: str | None = None) -> None:
    handlers = [logging.StreamHandler()]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(pref: str = "auto"):
    import torch

    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


@contextmanager
def timer(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = store.get(key, 0.0) + time.perf_counter() - t0


def dump_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class GPUEnergyMeter:
    """GPU energy (J) via NVML when available; None otherwise."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.energy_j = None
        self._ok = False
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nv = pynvml
            self._ok = True
        except Exception:
            pass

    def __enter__(self):
        if self._ok:
            try:
                self._e0 = self._nv.nvmlDeviceGetTotalEnergyConsumption(self._h)  # mJ
            except Exception:
                self._ok = False
        return self

    def __exit__(self, *exc):
        if self._ok:
            try:
                e1 = self._nv.nvmlDeviceGetTotalEnergyConsumption(self._h)
                self.energy_j = (e1 - self._e0) / 1000.0
            except Exception:
                self.energy_j = None
        return False


def env_snapshot() -> dict:
    """Software/hardware environment for reproducibility (written next to every run)."""
    import importlib
    import platform
    import subprocess
    import sys
    import time as _t

    info = {"timestamp": _t.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "argv": sys.argv}
    pkgs = {}
    for m in ["numpy", "pandas", "scipy", "sklearn", "imblearn", "xgboost", "optuna", "torch", "ctgan", "pyarrow"]:
        try:
            pkgs[m] = importlib.import_module(m).__version__
        except Exception:
            pkgs[m] = None
    info["packages"] = pkgs
    try:
        import torch

        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["gpus"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                         "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                         "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
                        for i in range(torch.cuda.device_count())]
        info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    except Exception:
        pass
    try:
        info["nvidia_driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                               capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        info["nvidia_driver"] = None
    try:
        root = Path(__file__).resolve().parents[1]
        info["git_commit"] = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                            text=True, timeout=10).stdout.strip() or None
    except Exception:
        info["git_commit"] = None
    try:
        info["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                            text=True, timeout=120).stdout.splitlines()
    except Exception:
        info["pip_freeze"] = None
    return info


def write_env_snapshot(out_dir) -> None:
    import time as _t

    dump_json(env_snapshot(), Path(out_dir) / f"environment_{_t.strftime('%Y%m%d_%H%M%S')}.json")
