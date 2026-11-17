"""Decode and resize every Waterbirds image once into a uint8 cache.

    python scripts/build_cache.py \
        --data_root /content/data/waterbird_complete95_forest2water2 \
        --out_dir /content/cache_256

Takes a few minutes on Colab. Build on local disk (memory-mapped writes to a
Drive mount are slow), copy the folder to Drive once, and copy it back to
local disk at the start of each session.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srl.cache import build_cache  # noqa: E402
from srl.data import load_pool, verify_official_counts  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    pool = load_pool(args.data_root)
    verify_official_counts(pool)
    build_cache(pool, args.data_root, args.out_dir, args.size, args.workers, args.overwrite)


if __name__ == "__main__":
    main()
