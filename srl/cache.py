"""Pre-resized uint8 image cache and an on-device image bank.

Why: on Colab the bottleneck is JPEG decoding on two CPU cores, not the GPU.
We decode and resize every image exactly once, store a uint8 array, and at
training time keep the images a run needs on the GPU, where cropping,
flipping and normalization are nearly free.

Preprocessing matches the Group DRO / DFR evaluation transform up to the
crop: resize to 256x256 (bilinear, aspect ratio not preserved), then a
224x224 center crop at evaluation time. Training augmentation reproduces
torchvision's RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(3/4, 4/3))
plus a horizontal flip, but applied to the 256x256 cached image (the
original transform crops the full-resolution image; the difference is a
small change in effective crop scale, identical for every method).
"""

from __future__ import annotations

import gc
import hashlib
import io
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .constants import IMAGENET_MEAN, IMAGENET_STD
from .utils import load_json, save_json, sha256_file

IMAGES_FILE = "images_{size}.npy"
INDEX_FILE = "index.csv"
MANIFEST_FILE = "manifest.json"
_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def _load_resize(job: tuple[str, int]):
    path, size = job
    with open(path, "rb") as f:
        raw = f.read()
    md5 = hashlib.md5(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        width, height = im.size
        arr = np.asarray(im.resize((size, size), _BILINEAR), dtype=np.uint8)
    return arr, width, height, md5


def build_cache(pool: pd.DataFrame, data_root: str | Path, out_dir: str | Path,
                size: int = 256, workers: int = 2, overwrite: bool = False) -> Path:
    """Decode, resize and store every image of the pool, in pool (img_id) order.

    Writes images_{size}.npy (N x size x size x 3 uint8), index.csv (metadata,
    original image sizes and an MD5 per file) and manifest.json (hashes and
    settings). Writing goes to a temporary file that is renamed at the end,
    so an interrupted build never leaves a valid-looking partial cache.
    """
    import PIL

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_path = out_dir / IMAGES_FILE.format(size=size)
    if images_path.exists() and (out_dir / MANIFEST_FILE).exists() and not overwrite:
        print(f"[cache] {images_path} already exists; pass overwrite=True to rebuild")
        return out_dir

    n = len(pool)
    partial = out_dir / f"images_{size}.partial.npy"
    mm = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=(n, size, size, 3))
    widths = np.zeros(n, dtype=np.int64)
    heights = np.zeros(n, dtype=np.int64)
    md5s = [""] * n
    jobs = [(str(Path(data_root) / fn), size) for fn in pool["img_filename"]]

    t0 = time.time()
    pool_ctx = Pool(workers) if workers > 1 else None
    try:
        results = pool_ctx.imap(_load_resize, jobs, chunksize=16) if pool_ctx else map(_load_resize, jobs)
        for i, (arr, w, h, md5) in enumerate(results):
            mm[i], widths[i], heights[i], md5s[i] = arr, w, h, md5
            if (i + 1) % 1000 == 0 or i + 1 == n:
                rate = (i + 1) / (time.time() - t0)
                print(f"[cache] {i + 1}/{n} images ({rate:.0f} img/s)")
    finally:
        if pool_ctx:
            pool_ctx.close()
            pool_ctx.join()
    mm.flush()
    del mm
    gc.collect()          # Windows: release the memory map before renaming the file
    os.replace(partial, images_path)

    keep = ["img_id", "img_filename", "y", "a", "g", "official_split"]
    if "place_filename" in pool.columns:
        keep.append("place_filename")
    index = pool[keep].copy()
    index["orig_width"], index["orig_height"], index["md5"] = widths, heights, md5s
    index_path = out_dir / INDEX_FILE
    index.to_csv(index_path, index=False)

    save_json(out_dir / MANIFEST_FILE, {
        "format_version": 1,
        "size": size,
        "n_images": n,
        "resize": "PIL bilinear to size x size (aspect ratio not preserved)",
        "pil_version": PIL.__version__,
        "numpy_version": np.__version__,
        "build_seconds": round(time.time() - t0, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "images_sha256": sha256_file(images_path),
        "index_sha256": sha256_file(index_path),
    })
    print(f"[cache] done in {time.time() - t0:.0f}s -> {out_dir}")
    return out_dir


class ImageCache:
    """Read-only, memory-mapped access to a cache built by build_cache."""

    def __init__(self, cache_dir: str | Path, verify: bool = False):
        self.dir = Path(cache_dir)
        if not (self.dir / MANIFEST_FILE).exists():
            raise FileNotFoundError(f"no cache manifest in {self.dir}; run scripts/build_cache.py")
        self.manifest = load_json(self.dir / MANIFEST_FILE)
        self.size = int(self.manifest["size"])
        images_path = self.dir / IMAGES_FILE.format(size=self.size)
        if verify and sha256_file(images_path) != self.manifest["images_sha256"]:
            raise ValueError(f"{images_path} does not match its manifest hash")
        self.index = pd.read_csv(self.dir / INDEX_FILE)
        self.images = np.load(images_path, mmap_mode="r")
        if self.images.shape[0] != len(self.index):
            raise ValueError("image array and index.csv disagree in length")
        self._ids = self.index["img_id"].to_numpy()
        if not np.all(np.diff(self._ids) > 0):
            raise ValueError("cache index must be strictly sorted by img_id")

    @property
    def fingerprint(self) -> str:
        return self.manifest["images_sha256"]

    def positions(self, img_ids) -> np.ndarray:
        img_ids = np.asarray(img_ids, dtype=np.int64)
        pos = np.searchsorted(self._ids, img_ids)
        bad = (pos >= len(self._ids)) | (self._ids[np.minimum(pos, len(self._ids) - 1)] != img_ids)
        if bad.any():
            raise KeyError(f"{int(bad.sum())} img_ids are not in the cache")
        return pos

    def take(self, img_ids) -> np.ndarray:
        """Images (N x H x W x 3 uint8) in the order of img_ids. Rows are read in
        sorted order, which is much faster on memory-mapped network storage."""
        pos = self.positions(img_ids)
        order = np.argsort(pos)
        out = np.empty((len(pos), self.size, self.size, 3), dtype=np.uint8)
        out[order] = self.images[pos[order]]
        return out

    def labels(self, img_ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self.index.iloc[self.positions(img_ids)]
        # copy=True: pandas copy-on-write can hand back read-only arrays
        return tuple(rows[c].to_numpy(np.int64, copy=True) for c in ("y", "a", "g"))


class DeviceImageBank:
    """A fixed set of uint8 images resident on one device that produces
    normalized float batches, cropping and augmenting on that device."""

    def __init__(self, images_nhwc: np.ndarray, device, crop: int = 224):
        import torch

        t = torch.from_numpy(np.ascontiguousarray(images_nhwc)).to(device)
        self.x = t.permute(0, 3, 1, 2).contiguous()          # N x 3 x H x W, uint8
        del t
        self.n, _, self.h, self.w = self.x.shape
        if crop > min(self.h, self.w):
            raise ValueError(f"crop {crop} larger than cached images {self.h}x{self.w}")
        self.crop = crop
        self.device = self.x.device
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1) * 255.0
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1) * 255.0

    def __len__(self) -> int:
        return self.n

    def batch(self, idx, augment: str = "none", gen=None,
              scale: tuple = (0.7, 1.0), ratio: tuple = (3 / 4, 4 / 3)):
        import torch

        x = self.x.index_select(0, idx)
        if augment == "none":
            top, left = (self.h - self.crop) // 2, (self.w - self.crop) // 2
            x = x[:, :, top:top + self.crop, left:left + self.crop].float()
        elif augment == "rrc_flip":
            x = self._random_resized_crop_flip(x.float(), gen, scale, ratio)
        else:
            raise ValueError(f"unknown augment {augment!r}")
        x = (x - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last)

    def _random_resized_crop_flip(self, x, gen, scale, ratio):
        """Vectorized RandomResizedCrop + horizontal flip via one affine grid_sample.

        Box parameters follow torchvision's sampling rule (10 attempts, then a
        whole-image fallback). With align_corners=False, a box with top-left
        pixel (i, j) and size (h, w) maps to normalized center
        c = (2j + w)/W - 1 and half-extent s = w/W, so output pixel centers
        land exactly on input pixel centers when no rescaling is needed.
        """
        import torch
        import torch.nn.functional as F

        b = x.shape[0]
        H, W = self.h, self.w
        attempts = 10
        target = (H * W) * torch.empty(b, attempts).uniform_(scale[0], scale[1], generator=gen)
        log_aspect = torch.empty(b, attempts).uniform_(math.log(ratio[0]), math.log(ratio[1]),
                                                       generator=gen)
        aspect = torch.exp(log_aspect)
        w = torch.round(torch.sqrt(target * aspect))
        h = torch.round(torch.sqrt(target / aspect))
        ok = (w > 0) & (h > 0) & (w <= W) & (h <= H)
        first = torch.argmax(ok.to(torch.int64), dim=1)
        rows = torch.arange(b)
        w, h = w[rows, first], h[rows, first]
        found = ok.any(dim=1)
        w = torch.where(found, w, torch.full_like(w, float(W)))
        h = torch.where(found, h, torch.full_like(h, float(H)))
        i = torch.floor(torch.rand(b, generator=gen) * (H - h + 1))
        j = torch.floor(torch.rand(b, generator=gen) * (W - w + 1))
        flip = torch.rand(b, generator=gen) < 0.5

        theta = torch.zeros(b, 2, 3)
        theta[:, 0, 0] = (w / W) * torch.where(flip, -1.0, 1.0)
        theta[:, 0, 2] = (2 * j + w) / W - 1
        theta[:, 1, 1] = h / H
        theta[:, 1, 2] = (2 * i + h) / H - 1
        grid = F.affine_grid(theta.to(x.device), size=(b, 3, self.crop, self.crop),
                             align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                             align_corners=False)
