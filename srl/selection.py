"""Model selection from logged validation logits (never from test).

Protocols (project plan, Section 6):
    S1  WGA on val_bal                 assumes group labels at validation time
    S2  average accuracy on val_bal    assumes an attribute-balanced validation set
    S3  worst-class accuracy on val_id  class labels only (Yang et al., 2023)
    S4  average accuracy on val_id      nothing beyond an ordinary held-out set

For the official split, S1 and S2 use the official validation set ('val'),
which is background-balanced within each class; S3 and S4 are undefined.

Ties are broken by lower NLL on the same validation set, then by the earlier
step. Step 0 (pretrained backbone, untrained head) is never a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import summarize

PROTOCOLS = {
    "S1": ("val_bal", "wga"),
    "S2": ("val_bal", "acc"),
    "S3": ("val_id", "worst_class_acc"),
    "S4": ("val_id", "acc"),
}
_ALIASES = {"val_bal": ("val_bal", "val")}


@dataclass
class Selection:
    protocol: str
    eval_set: str
    metric: str
    eval_index: int
    step: int
    score: float


def available_protocols(run_dir: str | Path) -> list[str]:
    return [p for p in PROTOCOLS if _find_set(Path(run_dir), PROTOCOLS[p][0]) is not None]


def _find_set(run_dir: Path, name: str) -> str | None:
    for candidate in _ALIASES.get(name, (name,)):
        if (run_dir / "logits" / f"{candidate}.npy").exists():
            return candidate
    return None


def validation_curve(run_dir: str | Path, protocol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Selection score and NLL at every logged step for one protocol."""
    run_dir = Path(run_dir)
    set_name, metric = PROTOCOLS[protocol]
    found = _find_set(run_dir, set_name)
    if found is None:
        raise FileNotFoundError(f"{run_dir.name}: no logits for {set_name}; protocol {protocol} undefined")
    steps = np.load(run_dir / "logits" / "eval_steps.npy")
    logits = np.load(run_dir / "logits" / f"{found}.npy")
    sets = np.load(run_dir / "sets.npz")
    y, g = sets[f"{found}_y"], sets[f"{found}_g"]
    scores, nlls = np.empty(len(steps)), np.empty(len(steps))
    for i in range(len(steps)):
        m = summarize(logits[i], y, g)
        scores[i], nlls[i] = m[metric], m["nll"]
    return steps, scores, nlls, found


def select(run_dir: str | Path, protocol: str) -> Selection:
    steps, scores, nlls, found = validation_curve(run_dir, protocol)
    candidates = np.flatnonzero(steps > 0)
    best = scores[candidates].max()
    tied = candidates[np.isclose(scores[candidates], best)]
    i = int(tied[np.lexsort((steps[tied], nlls[tied]))[0]])
    return Selection(protocol, found, PROTOCOLS[protocol][1], i, int(steps[i]), float(scores[i]))
