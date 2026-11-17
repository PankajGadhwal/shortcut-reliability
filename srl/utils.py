"""Small utilities shared across the package: hashing, atomic writes, seeding,
determinism flags and environment capture for reproducibility records."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


def rho_key(rho: float) -> str:
    """Canonical string for a correlation strength: 0.95 -> '0.95', 0.5 -> '0.5'."""
    return f"{float(rho):.4f}".rstrip("0").rstrip(".")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Hash of a JSON-serializable object, independent of key order."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write via a temporary file and rename, so an interrupted Colab session
    never leaves a half-written file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def save_json(path: str | Path, obj: Any, compact: bool = False) -> None:
    if compact:   # large id lists: one line, about a third of the size
        text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    else:
        text = json.dumps(obj, indent=2, sort_keys=True, default=str)
    atomic_write_text(path, text)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def git_info(repo_dir: str | Path | None = None) -> dict:
    repo_dir = Path(repo_dir or Path(__file__).resolve().parents[1])
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                                capture_output=True, text=True, timeout=10).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir,
                                capture_output=True, text=True, timeout=10).stdout.strip()
        return {"commit": commit or None, "dirty": bool(status)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def set_global_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(deterministic: bool) -> None:
    """cuDNN flags. Bitwise reproducibility on GPU is still not guaranteed
    (some backward kernels, e.g. adaptive average pooling, use atomic adds),
    which is one more reason the study measures variability across seeds."""
    import torch

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def env_info() -> dict:
    import torch
    import torchvision

    info = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git": git_info(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cudnn"] = torch.backends.cudnn.version()
    return info
