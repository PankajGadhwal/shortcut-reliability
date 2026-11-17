"""Build the controlled splits once, validate every invariant, and save them.

    python scripts/make_splits.py --data_root /content/data/waterbird_complete95_forest2water2

Commit the resulting JSON to git: it is part of the experimental record.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from srl.data import (SplitSpec, background_overlap, load_pool, make_controlled_splits,  # noqa: E402
                      save_splits, validate_controlled_splits, verify_official_counts)
from srl.utils import sha256_file  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out", default="splits/controlled_v1.json")
    p.add_argument("--rhos", type=float, nargs="+", default=list(SplitSpec.rhos))
    p.add_argument("--data_seeds", type=int, nargs="+", default=list(SplitSpec.data_seeds))
    p.add_argument("--split_seed", type=int, default=SplitSpec.split_seed)
    p.add_argument("--test_per_group", type=int, default=SplitSpec.test_per_group)
    p.add_argument("--val_bal_per_group", type=int, default=SplitSpec.val_bal_per_group)
    p.add_argument("--train_per_class", type=int, default=SplitSpec.train_per_class)
    p.add_argument("--val_id_per_class", type=int, default=SplitSpec.val_id_per_class)
    p.add_argument("--name", default=SplitSpec.name)
    p.add_argument("--allow_count_mismatch", action="store_true",
                   help="warn instead of failing if the download differs from the official counts")
    args = p.parse_args(argv)

    pool = load_pool(args.data_root)
    table = verify_official_counts(pool, strict=not args.allow_count_mismatch)
    print("Official split counts (observed vs expected):")
    print(table.to_string(index=False), "\n")

    spec = SplitSpec(rhos=tuple(args.rhos), data_seeds=tuple(args.data_seeds),
                     split_seed=args.split_seed, test_per_group=args.test_per_group,
                     val_bal_per_group=args.val_bal_per_group,
                     train_per_class=args.train_per_class,
                     val_id_per_class=args.val_id_per_class, name=args.name)
    splits = make_controlled_splits(pool, spec)
    splits["metadata_sha256"] = sha256_file(Path(args.data_root) / "metadata.csv")
    summary = validate_controlled_splits(pool, splits)
    print("Validation passed: counts, disjointness and nesting hold for every seed and rho.\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary[summary.seed == summary.seed.min()].to_string(index=False), "\n")

    test = splits["fixed"]["test"]
    all_train = sorted({i for by_rho in splits["draws"].values() for d in by_rho.values()
                        for i in d["train"] + d["val_id"]})
    overlap = background_overlap(pool, all_train, test)
    if overlap is None:
        print("metadata.csv has no place_filename column: background overlap not checked")
    else:
        print(f"Background images shared between any train/val_id set and test: {overlap}")

    sha = save_splits(args.out, splits)
    print(f"\nSaved {args.out}\nsha256 {sha}")


if __name__ == "__main__":
    main()
