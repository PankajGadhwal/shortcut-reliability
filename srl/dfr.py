"""Deep Feature Reweighting (Kirichenko et al., 2023): keep the ERM feature
extractor and refit only the last layer, as an L1-regularized logistic
regression on group-balanced held-out data.

Protocol (the paper's validation-based variant, adapted to our sets):
  1. The held-out set is val_bal (controlled split) or the official val set.
  2. Split it in half, stratified by group. For each C in the grid, fit on
     half A and score worst-group accuracy on half B. Each fit averages the
     coefficients of `n_avg` heads, each trained on a group-balanced random
     subsample (all groups cut to the size of the smallest), standardized
     with statistics of that training data.
  3. Refit with the chosen C on the whole held-out set the same way.

DFR therefore consumes the group labels of the held-out set: it is defined
only under protocol S1, and its output run says so. Its in-sample val_bal
predictions are useless for model selection; the honest validation score is
the half-B worst-group accuracy stored as `dfr_tuning`.

The result is written as a run directory (method "dfr") with a single
"evaluation step", so selection and final evaluation treat it like any run.
Sealed test embeddings are read only to write sealed test logits; nothing is
scored on them here.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from .metrics import group_accuracies
from .utils import load_json, save_json

C_GRID = (1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01)


def _logreg(C: float):
    import sklearn
    from sklearn.linear_model import LogisticRegression

    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):        # 'penalty' is deprecated from 1.8 on
        return LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear")
    return LogisticRegression(C=C, penalty="l1", solver="liblinear")


def balanced_subsample(g: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups = np.unique(g)
    m = min(int((g == k).sum()) for k in groups)
    return np.concatenate([rng.choice(np.flatnonzero(g == k), m, replace=False) for k in groups])


def fit_head(X, y, g, C, rng, n_avg=10):
    """Average of n_avg standardized L1 logistic heads on balanced subsamples.
    Returns (w, b) acting on raw features: logit = X @ w + b."""
    ws, bs = [], []
    for _ in range(n_avg):
        idx = balanced_subsample(g, rng)
        mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-8
        clf = _logreg(C).fit((X[idx] - mu) / sd, y[idx])
        w = clf.coef_[0] / sd
        ws.append(w)
        bs.append(clf.intercept_[0] - float(mu @ w))
    return np.mean(ws, axis=0), float(np.mean(bs))


def two_class_logits(X, w, b) -> np.ndarray:
    z = X @ w + b
    return np.stack([np.zeros_like(z), z], axis=1)     # softmax([0, z]) = sigmoid(z)


def stratified_halves(g: np.ndarray, rng: np.random.Generator):
    a, b = [], []
    for k in np.unique(g):
        idx = rng.permutation(np.flatnonzero(g == k))
        a.append(idx[: len(idx) // 2])
        b.append(idx[len(idx) // 2:])
    return np.concatenate(a), np.concatenate(b)


def run_dfr(erm_run: str | Path, out_dir: str | Path | None = None, c_grid=C_GRID,
            n_avg: int = 10, seed: int | None = None, tag: str = "") -> Path:
    erm_run = Path(erm_run)
    cfg = load_json(erm_run / "config.json")
    if cfg["method"] != "erm" or not (erm_run / "DONE").exists():
        raise ValueError(f"{erm_run.name} is not a finished ERM run")
    name = erm_run.name.replace("_erm_", "_dfr_", 1) + (f"_{tag}" if tag else "")
    out = Path(out_dir or erm_run.parent) / name
    if (out / "DONE").exists():
        print(f"[skip] {out.name} is already complete")
        return out
    seed = cfg["train_seed"] if seed is None else seed
    rng = np.random.default_rng([seed, 2023])

    sets = np.load(erm_run / "sets.npz")
    held = "val_bal" if (erm_run / "embeddings" / "val_bal.npy").exists() else "val"
    X = np.load(erm_run / "embeddings" / f"{held}.npy").astype(np.float64)
    y, g = sets[f"{held}_y"], sets[f"{held}_g"]

    half_a, half_b = stratified_halves(g, rng)
    scores = []
    for C in c_grid:
        w, b = fit_head(X[half_a], y[half_a], g[half_a], C, rng, n_avg)
        pred = two_class_logits(X[half_b], w, b).argmax(1)
        scores.append(float(np.nanmin(group_accuracies(pred == y[half_b], g[half_b]))))
    best = int(np.argmax(scores))                       # ties -> larger C (first in grid)
    w, b = fit_head(X, y, g, c_grid[best], rng, n_avg)

    out.mkdir(parents=True, exist_ok=True)
    (out / "logits").mkdir(exist_ok=True)
    (out / "sealed").mkdir(exist_ok=True)
    step = int(np.load(erm_run / "logits" / "eval_steps.npy")[-1])
    np.save(out / "logits" / "eval_steps.npy", np.array([step]))
    for f in sorted((erm_run / "embeddings").glob("*.npy")):
        feats = np.load(f).astype(np.float64)
        np.save(out / "logits" / f.name, two_class_logits(feats, w, b)[None].astype(np.float16))
    test_feats = np.load(erm_run / "sealed" / "test_embeddings.npy").astype(np.float64)
    np.save(out / "sealed" / "test.npy", two_class_logits(test_feats, w, b)[None].astype(np.float16))
    shutil.copy(erm_run / "sets.npz", out / "sets.npz")
    shutil.copy(erm_run / "summary.json", out / "summary.json")
    np.savez(out / "dfr_head.npz", w=w, b=b)
    save_json(out / "config.json", {
        **cfg, "method": "dfr", "protocols": ["S1"], "source_run": erm_run.name,
        "dfr": {"held_out_set": held, "c_grid": list(c_grid), "half_b_wga": scores,
                "chosen_C": c_grid[best], "n_avg": n_avg, "seed": seed,
                "nonzero_weights": int((np.abs(w) > 0).sum()), "n_features": int(len(w))},
        "dfr_tuning": {"metric": "worst-group accuracy on half B", "score": scores[best]}})
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(f"[dfr] {out.name}: C={c_grid[best]} (half-B WGA {scores[best]:.3f}); "
          f"{int((np.abs(w) > 0).sum())}/{len(w)} nonzero weights")
    return out
