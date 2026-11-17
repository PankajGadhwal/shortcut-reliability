"""Evaluation metrics. Everything is computed from logits, labels and groups,
so any metric can be recomputed later from the logged logits.

Key identity (project plan, Section 6). For a class-balanced distribution in
which a fraction rho' of each class has the agreeing background,

    Acc(rho') = rho' * Acc_maj + (1 - rho') * Acc_min,

where Acc_maj (Acc_min) is the class-average accuracy on the majority
(minority) groups. Hence:
    rho' = rho_train -> in-distribution accuracy (class-balanced training)
    rho' = 0.5       -> mean group accuracy
    rho' = 0         -> shortcut fully reversed
and the shortcut gap Gamma = Acc_maj - Acc_min is the accuracy lost per unit
of shift. The identity relies on the subpopulation-shift assumption that the
group-conditional distributions are the same in every split, which holds by
construction in the controlled re-splits.
"""

from __future__ import annotations

import numpy as np

from .constants import MAJORITY_GROUPS, MINORITY_GROUPS, N_CLASSES, N_GROUPS


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def group_accuracies(correct: np.ndarray, g: np.ndarray, n_groups: int = N_GROUPS) -> np.ndarray:
    """Accuracy per group (NaN for groups absent from the set)."""
    totals = np.bincount(g, minlength=n_groups).astype(float)
    hits = np.bincount(g, weights=correct.astype(float), minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, hits / totals, np.nan)


def class_accuracies(correct: np.ndarray, y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    return group_accuracies(correct, y, n_classes)


def majority_minority(group_accs: np.ndarray) -> tuple[float, float]:
    acc_maj = float(np.nanmean(group_accs[list(MAJORITY_GROUPS)]))
    acc_min = float(np.nanmean(group_accs[list(MINORITY_GROUPS)]))
    return acc_maj, acc_min


def acc_at_rho(group_accs: np.ndarray, rho_test: float) -> float:
    """Accuracy on a class-balanced test distribution with correlation rho_test."""
    acc_maj, acc_min = majority_minority(group_accs)
    return rho_test * acc_maj + (1.0 - rho_test) * acc_min


def mixture_accuracy(group_accs: np.ndarray, weights) -> float:
    """Accuracy under arbitrary group proportions, e.g. the training mixture.
    This is how the Waterbirds literature defines 'average accuracy' on the
    official test set, whose group proportions differ from training."""
    w = np.asarray(weights, dtype=float)
    return float(np.nansum(w * group_accs) / w[~np.isnan(group_accs)].sum())


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10,
                               scheme: str = "equal_mass") -> float:
    """Binned ECE. Equal-mass bins by default: with a few hundred examples per
    group, equal-width bins leave most bins nearly empty and the estimate
    becomes noisier and more biased. Report NLL and Brier score alongside."""
    n = len(confidence)
    if n == 0:
        return float("nan")
    if scheme == "equal_mass":
        order = np.argsort(confidence, kind="stable")
        bins = np.array_split(order, min(n_bins, n))
    elif scheme == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        which = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
        bins = [np.flatnonzero(which == b) for b in range(n_bins)]
    else:
        raise ValueError(f"unknown ECE scheme {scheme!r}")
    ece = 0.0
    for b in bins:
        if len(b):
            ece += len(b) / n * abs(correct[b].mean() - confidence[b].mean())
    return float(ece)


def summarize(logits, y, g, train_group_props=None, ece_bins: int = 10) -> dict:
    """All scalar metrics for one set of predictions."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    g = np.asarray(g, dtype=np.int64)
    logp = log_softmax(logits)
    p = np.exp(logp)
    pred = p.argmax(axis=1)
    correct = pred == y
    confidence = p.max(axis=1)
    ga = group_accuracies(correct, g)
    acc_maj, acc_min = majority_minority(ga)
    onehot = np.eye(logits.shape[1])[y]

    out = {
        "n": int(len(y)),
        "acc": float(correct.mean()),
        "wga": float(np.nanmin(ga)),
        "worst_group": int(np.nanargmin(ga)),
        "mean_group_acc": float(np.nanmean(ga)),
        "worst_class_acc": float(np.nanmin(class_accuracies(correct, y))),
        "acc_maj": acc_maj,
        "acc_min": acc_min,
        "shortcut_gap": acc_maj - acc_min,
        "nll": float(-logp[np.arange(len(y)), y].mean()),
        "brier": float(((p - onehot) ** 2).sum(axis=1).mean()),
        "ece": expected_calibration_error(confidence, correct, ece_bins),
    }
    group_ece = []
    for k in range(N_GROUPS):
        mask = g == k
        out[f"acc_g{k}"] = float(ga[k])
        out[f"n_g{k}"] = int(mask.sum())
        e = expected_calibration_error(confidence[mask], correct[mask], ece_bins) if mask.any() else np.nan
        out[f"ece_g{k}"] = e
        group_ece.append(e)
    out["worst_group_ece"] = float(np.nanmax(group_ece))
    if train_group_props is not None:
        out["id_acc"] = mixture_accuracy(ga, train_group_props)
    return out
