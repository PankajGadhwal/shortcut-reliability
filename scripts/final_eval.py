"""Model selection report for finished runs; test metrics only with --unseal.

Safe mode (default) prints, for each protocol, the step selected on
validation data. With --unseal it also reads the sealed test logits at those
steps. For the controlled study, use --unseal only after the preregistration
is committed. The official-split anchor check is the one earlier exception.

    python scripts/final_eval.py --runs "/content/drive/MyDrive/es678/runs/official_*"
    python scripts/final_eval.py --runs "/content/drive/MyDrive/es678/runs/official_*" --unseal
"""

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from srl.evaluate import evaluate_run  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="glob of run directories")
    p.add_argument("--protocols", nargs="+", default=None)
    p.add_argument("--unseal", action="store_true")
    p.add_argument("--out_csv", default=None)
    args = p.parse_args(argv)
    run_dirs = sorted(d for d in glob.glob(args.runs) if (Path(d) / "DONE").exists())
    if not run_dirs:
        sys.exit(f"no finished runs match {args.runs}")
    rows = [r for d in run_dirs for r in evaluate_run(d, args.protocols, unseal=args.unseal)]
    df = pd.DataFrame(rows)
    cols = ["run", "protocol", "selected_step", "val_set", "val_score"]
    if args.unseal:
        cols += ["test_id_acc", "test_acc", "test_wga", "test_mean_group_acc", "test_shortcut_gap",
                 "test_ece", "test_worst_group_ece"]
    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.float_format", "{:.4f}".format):
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"saved {args.out_csv}")


if __name__ == "__main__":
    main()
