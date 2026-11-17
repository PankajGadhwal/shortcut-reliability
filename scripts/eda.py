"""Exploratory data analysis and integrity checks. Writes a markdown report
and figures to --out_dir. Needs only the cache and the splits file.

    python scripts/eda.py --cache_dir /content/cache_256 \
        --splits splits/controlled_v1.json --out_dir reports/eda
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from srl.cache import ImageCache  # noqa: E402
from srl.constants import GROUP_NAMES  # noqa: E402
from srl.data import (background_overlap, load_splits, official_count_table,  # noqa: E402
                      validate_controlled_splits)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out_dir", default="reports/eda")
    p.add_argument("--samples_per_group", type=int, default=8)
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache = ImageCache(args.cache_dir)
    idx = cache.index
    splits, splits_sha = load_splits(args.splits)
    report = [f"# EDA report\n\nCache images sha256 `{cache.fingerprint[:16]}`, "
              f"splits sha256 `{splits_sha[:16]}`, {len(idx)} images.\n"]

    # 1. Official counts
    official = official_count_table(idx)
    official.to_csv(out / "official_counts.csv", index=False)
    report += ["## 1. Official split counts\n", md_table(official),
               f"\nAll match the official release: **{bool(official['match'].all())}**\n"]

    # 2. Pool contingency
    pool = pd.crosstab(idx["y"].map({0: "landbird", 1: "waterbird"}),
                       idx["a"].map({0: "land", 1: "water"}))
    report += ["## 2. Whole pool (label x background)\n", md_table(pool.reset_index()), ""]

    # 3. Controlled design
    summary = validate_controlled_splits(idx, splits)
    summary.to_csv(out / "controlled_summary.csv", index=False)
    s0 = summary[summary.seed == summary.seed.min()]
    keep = ["rho", "train_minority_per_class", "realized_rho_landbirds",
            "realized_rho_waterbirds"] + [c for c in s0.columns if c.startswith("val_id_")]
    report += ["## 3. Controlled splits (validated: counts, disjointness, nesting)\n",
               f"Fixed test: {len(splits['fixed']['test'])} images; fixed val_bal: "
               f"{len(splits['fixed']['val_bal'])}. Seeds: {list(splits['draws'])}. "
               "Per-seed rows are in controlled_summary.csv; first seed shown.\n",
               md_table(s0[keep]), ""]

    # 4. Integrity: background reuse and exact duplicate files
    test_ids = splits["fixed"]["test"]
    trainish = sorted({i for by_rho in splits["draws"].values() for d in by_rho.values()
                       for i in d["train"] + d["val_id"]})
    ov_test = background_overlap(idx, trainish, test_ids)
    ov_val = background_overlap(idx, trainish, splits["fixed"]["val_bal"])
    dup = idx[idx.duplicated("md5", keep=False)]
    set_of = {i: "test" for i in test_ids}
    set_of.update({i: "val_bal" for i in splits["fixed"]["val_bal"]})
    cross = 0
    for _, grp in dup.groupby("md5"):
        if len({set_of.get(i, "train-pool") for i in grp["img_id"]}) > 1:
            cross += 1
    report += ["## 4. Integrity checks\n",
               f"- Background images shared between any train/val_id set and test: {ov_test}",
               f"- ... and between any train/val_id set and val_bal: {ov_val}",
               f"- Files with byte-identical duplicates: {len(dup)} "
               f"({dup['md5'].nunique() if len(dup) else 0} distinct contents); "
               f"duplicate groups spanning test/val_bal/train pool: {cross}", ""]

    # 5. Original image sizes
    sizes = idx.groupby("g")[["orig_width", "orig_height"]].median().reset_index()
    sizes["group"] = [GROUP_NAMES[g] for g in sizes["g"]]
    report += ["## 5. Original image sizes (median per group)\n",
               md_table(sizes[["group", "orig_width", "orig_height"]]), ""]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].hist(idx["orig_width"], bins=40)
    ax[0].set_title("original width (px)")
    ax[1].hist(idx["orig_height"], bins=40)
    ax[1].set_title("original height (px)")
    fig.tight_layout()
    fig.savefig(out / "image_sizes.png", dpi=120)
    plt.close(fig)

    # 6. Sample grid from the fixed test set
    rng = np.random.default_rng(0)
    n = args.samples_per_group
    fig, axes = plt.subplots(4, n, figsize=(1.6 * n, 7))
    test_df = idx.set_index("img_id").loc[test_ids].reset_index()
    for g in range(4):
        pick = rng.choice(test_df.loc[test_df.g == g, "img_id"].to_numpy(), n, replace=False)
        for k, im in enumerate(cache.take(pick)):
            axes[g, k].imshow(im)
            axes[g, k].axis("off")
        axes[g, 0].set_title(GROUP_NAMES[g], loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "group_samples.png", dpi=120)
    plt.close(fig)
    report += ["## 6. Figures\n", "![sizes](image_sizes.png)\n", "![samples](group_samples.png)\n"]

    (out / "eda_report.md").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nwrote {out / 'eda_report.md'}")


if __name__ == "__main__":
    main()
