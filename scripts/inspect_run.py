"""Inspect one run using training and validation data only (never test).

Saves <run>/curves.png and prints timing plus a projection of what a full
run and the main grid will cost on this machine.

    python scripts/inspect_run.py runs/<run_name>
    python scripts/inspect_run.py runs/<run_name> --full_steps 2000 --methods 4 --rhos 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from srl.train import TrainConfig, make_eval_steps  # noqa: E402
from srl.utils import load_json  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--full_steps", type=int, default=2000)
    p.add_argument("--methods", type=int, default=4)
    p.add_argument("--rhos", type=int, default=5)
    args = p.parse_args(argv)
    run = Path(args.run_dir)
    log = pd.read_csv(run / "train_log.csv")

    # ---- curves (train and validation columns only) ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for col in log.columns:
        if col.startswith(("train_", "val")) and col.endswith(("_acc", "_wga")) and "loss" not in col:
            ax[0].plot(log["step"], log[col], marker=".", label=col)
    ax[0].set_xlabel("step")
    ax[0].set_ylim(0, 1.02)
    ax[0].legend(fontsize=7)
    ax[0].set_title(f"{run.name}\ntraining and validation (no test)", fontsize=8)
    ax[1].plot(log["step"], log["train_loss"], marker=".")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("step")
    ax[1].set_title("training loss")
    fig.tight_layout()
    fig.savefig(run / "curves.png", dpi=120)
    print(f"saved {run / 'curves.png'}")

    last = log.iloc[-1]
    print("\nfinal step:", int(last["step"]))
    for col in [c for c in log.columns if c.endswith(("_acc", "_wga"))]:
        print(f"  {col:28s} {last[col]:.3f}")

    # ---- timing and projection ----
    if not (run / "summary.json").exists():
        print("\nrun not finished: no timing summary yet")
        return
    s = load_json(run / "summary.json")
    cfg = load_json(run / "config.json")
    sec_per_step = s["train_seconds"] / cfg["steps"]
    sec_per_eval = s["eval_seconds"] / s["n_evals"]
    d = TrainConfig()     # the full runs use the default evaluation schedule
    n_evals_full = len(make_eval_steps(args.full_steps, d.eval_every, d.dense_eval_every,
                                       d.dense_eval_until))
    per_run = args.full_steps * sec_per_step + n_evals_full * sec_per_eval
    print(f"\ntraining {s['train_images_per_second']:.0f} img/s; {sec_per_eval:.1f} s per evaluation; "
          f"peak GPU memory {s.get('max_gpu_memory_gb')} GB")
    print(f"projected full run ({args.full_steps} steps, {n_evals_full} evaluations): {per_run / 60:.1f} min")
    for seeds in (3, 5):
        n = args.rhos * args.methods * seeds
        print(f"main grid, {seeds} seeds ({n} runs): {n * per_run / 3600:.1f} GPU-hours")


if __name__ == "__main__":
    main()
