"""Final evaluation: the only code that reads sealed test outputs.

For each selection protocol, the checkpoint is chosen from validation logits
(srl.selection) and only then is the test row at that index read. Reading
test outputs requires unseal=True. That flag is a speed bump, not security:
for the controlled study, set it only after the preregistration is
committed and the protocol frozen. The one legitimate earlier use is the
official-split anchor check of week 2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics import acc_at_rho, summarize
from .selection import available_protocols, select
from .utils import load_json, save_json


def evaluate_run(run_dir: str | Path, protocols=None, unseal: bool = False) -> list[dict]:
    run_dir = Path(run_dir)
    if not (run_dir / "DONE").exists():
        raise RuntimeError(f"{run_dir.name} is not complete")
    protocols = protocols or available_protocols(run_dir)
    cfg = load_json(run_dir / "config.json")
    summary = load_json(run_dir / "summary.json")
    rows = []
    for p in protocols:
        sel = select(run_dir, p)
        row = {"run": run_dir.name, "split": cfg["split"], "rho": cfg.get("rho"),
               "data_seed": cfg["data_seed"], "train_seed": cfg["train_seed"],
               "method": cfg["method"], "arch": cfg["arch"], "protocol": p,
               "selected_step": sel.step, "val_set": sel.eval_set, "val_score": sel.score}
        if unseal:
            sets = np.load(run_dir / "sets.npz")
            test_logits = np.load(run_dir / "sealed" / "test.npy", mmap_mode="r")[sel.eval_index]
            m = summarize(np.asarray(test_logits), sets["test_y"], sets["test_g"],
                          train_group_props=summary["train_group_props"])
            row.update({f"test_{k}": v for k, v in m.items()})
            if cfg["split"] == "controlled":
                ga = np.array([m[f"acc_g{k}"] for k in range(4)])
                for r in (0.0, 0.5, float(cfg["rho"])):
                    row[f"test_acc_at_rho_{r:g}"] = acc_at_rho(ga, r)
        rows.append(row)
    if unseal:
        save_json(run_dir / "final_eval.json", rows)
    return rows
