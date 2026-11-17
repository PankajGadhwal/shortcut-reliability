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
    # One panel per role, explicit colours/line styles: the default colour cycle
    # has 10 colours, so a single panel with 12 lines reuses colours.
    panels = [
        ("training set (eval mode)", [("train_acc", "accuracy", "C0", "-"),
                                      ("train_wga", "worst-group acc", "C3", "-")]),
        ("validation", [("val_bal_acc", "val_bal accuracy (S2)", "C0", "-"),
                        ("val_bal_wga", "val_bal worst-group (S1)", "C3", "-"),
                        ("val_bal_worst_class_acc", "val_bal worst-class", "C1", ":"),
                        ("val_id_acc", "val_id accuracy (S4)", "C2", "--"),
                        ("val_id_worst_class_acc", "val_id worst-class (S3)", "C4", "--"),
                        ("val_acc", "val accuracy (S2)", "C0", "-"),
                        ("val_wga", "val worst-group (S1)", "C3", "-")]),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for a, (title, series) in zip(ax[:2], panels):
        for col, label, color, style in series:
            if col in log.columns:
                a.plot(log["step"], log[col], style, color=color, marker=".", ms=3, label=label)
        a.set_xlabel("step")
        a.set_ylim(0, 1.02)
        a.grid(alpha=0.3)
        a.legend(fontsize=7, loc="lower right")
        a.set_title(title, fontsize=9)
    ax[2].plot(log["step"], log["train_loss"], marker=".", color="k")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("step")
    ax[2].grid(alpha=0.3)
    ax[2].set_title("training loss (mean since previous evaluation)", fontsize=9)
    fig.suptitle(f"{run.name}  (no test data)", fontsize=9)
    fig.tight_layout()
    fig.savefig(run / "curves.png", dpi=130)
    plt.close(fig)
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
