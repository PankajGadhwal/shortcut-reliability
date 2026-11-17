"""Validation-only comparison of finished runs, grouped by configuration and
averaged over data seeds. Never reads test outputs: use it for pilots and
hyperparameter decisions.

    python scripts/summarize_validation.py --runs "runs/controlled_rho0.95_*"

Columns (mean +- std over seeds):
  S1_wga      best val_bal worst-group accuracy over checkpoints (selection score)
  S1_step     step selected by S1
  S4_val_id   best val_id accuracy over checkpoints
  final_wga   val_bal worst-group accuracy at the last step
For DFR, S1_wga is the half-B tuning score (its val_bal predictions are in-sample).
"""

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from srl.selection import select  # noqa: E402
from srl.utils import load_json  # noqa: E402


def label(cfg: dict) -> str:
    parts = [cfg["method"], f"wd={cfg['weight_decay']:g}"]
    if cfg["method"] == "gdro":
        parts.append(f"eta_q={cfg['eta_q']:g}")
    if cfg["method"] == "jtt":
        parts += [f"T={cfg['jtt_T']}", f"lam={cfg['jtt_lambda']}"]
    if cfg.get("tag"):
        parts.append(cfg["tag"])
    return " ".join(parts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--out_csv", default=None)
    args = p.parse_args(argv)
    rows = []
    for d in sorted(glob.glob(args.runs)):
        d = Path(d)
        if not (d / "DONE").exists():
            continue
        cfg = load_json(d / "config.json")
        row = {"config": label(cfg), "rho": cfg.get("rho"), "seed": cfg["data_seed"], "run": d.name}
        if cfg["method"] == "dfr":
            row.update(S1_wga=cfg["dfr_tuning"]["score"], S1_step=float("nan"),
                       S4_val_id=float("nan"), final_wga=float("nan"))
        else:
            s1 = select(d, "S1")
            row.update(S1_wga=s1.score, S1_step=s1.step)
            try:
                row["S4_val_id"] = select(d, "S4").score
            except FileNotFoundError:
                row["S4_val_id"] = float("nan")
            log = pd.read_csv(d / "train_log.csv")
            wga_col = "val_bal_wga" if "val_bal_wga" in log else "val_wga"
            row["final_wga"] = float(log[wga_col].iloc[-1])
        if (d / "jtt_error_set.json").exists():
            a = load_json(d / "jtt_error_set.json")
            row.update(err_size=a["error_set_size"], err_precision=a["precision_minority"],
                       err_recall=a["recall_minority"])
        rows.append(row)
    if not rows:
        sys.exit(f"no finished runs match {args.runs}")
    df = pd.DataFrame(rows)
    metrics = [c for c in ("S1_wga", "S1_step", "S4_val_id", "final_wga", "err_size",
                           "err_precision", "err_recall") if c in df]
    agg = df.groupby(["rho", "config"])[metrics].agg(["mean", "std"])
    agg.insert(0, ("n_seeds", ""), df.groupby(["rho", "config"]).size())
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.float_format", "{:.3f}".format):
        print(agg)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nper-run rows saved to {args.out_csv}")


if __name__ == "__main__":
    main()
