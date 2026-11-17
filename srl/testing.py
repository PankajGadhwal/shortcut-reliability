"""Synthetic stand-ins for Waterbirds, used ONLY by the test suite and for
CPU dry runs of the pipeline. Nothing produced from these files is a result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .constants import EXPECTED_OFFICIAL_COUNTS, OFFICIAL_SPLIT_CODES


def fake_metadata(counts_per_split: dict | None = None, seed: int = 0) -> pd.DataFrame:
    """metadata.csv-like table. Defaults to the exact official counts."""
    counts_per_split = counts_per_split or EXPECTED_OFFICIAL_COUNTS
    rng = np.random.default_rng(seed)
    rows, img_id = [], 1
    for split, counts in counts_per_split.items():
        for g, n in enumerate(counts):
            for _ in range(n):
                y, a = divmod(g, 2)
                rows.append({"img_id": img_id, "img_filename": f"fake/{img_id:05d}.jpg", "y": y,
                             "split": OFFICIAL_SPLIT_CODES[split], "place": a,
                             "place_filename": f"/place/{img_id:05d}.jpg"})
                img_id += 1
    df = pd.DataFrame(rows)
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)   # scrambled order on purpose


def write_fake_dataset(root: str | Path, counts_per_split: dict, size=(40, 30), seed: int = 0) -> Path:
    """Tiny JPEGs whose background colour encodes a and a central patch encodes y."""
    root = Path(root)
    meta = fake_metadata(counts_per_split, seed)
    rng = np.random.default_rng(seed)
    w, h = size
    for row in meta.itertuples():
        bg = np.array([60, 140, 60]) if row.place == 0 else np.array([50, 90, 170])
        img = np.clip(bg + rng.normal(0, 25, (h, w, 3)), 0, 255)
        img[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = [220, 200, 40] if row.y == 0 else [230, 230, 230]
        path = root / row.img_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img.astype(np.uint8)).save(path, quality=90)
    meta.to_csv(root / "metadata.csv", index=False)
    return root
