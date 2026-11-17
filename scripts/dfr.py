"""Fit DFR heads on finished ERM runs (see srl/dfr.py for the protocol).

    python scripts/dfr.py --runs "runs/controlled_rho0.95_ds*_erm_resnet50_*"

Each ERM run produces a sibling run directory with '_erm_' replaced by '_dfr_'.
Reruns skip finished DFR runs.
"""

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srl.dfr import C_GRID, run_dfr  # noqa: E402
from srl.utils import load_json  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="glob of ERM run directories")
    p.add_argument("--c_grid", type=float, nargs="+", default=list(C_GRID))
    p.add_argument("--n_avg", type=int, default=10)
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)
    runs = [d for d in sorted(glob.glob(args.runs))
            if (Path(d) / "DONE").exists() and load_json(Path(d) / "config.json")["method"] == "erm"]
    if not runs:
        sys.exit(f"no finished ERM runs match {args.runs}")
    for d in runs:
        run_dfr(d, c_grid=tuple(args.c_grid), n_avg=args.n_avg, tag=args.tag)


if __name__ == "__main__":
    main()
