"""Training objectives. Each method turns per-example losses into the scalar
that is backpropagated, declares the sampler it needs, and carries any state
(Group DRO's group weights) through checkpoints. Every method shares the same
data pipeline, optimizer, step budget and evaluation schedule.

    erm   mean loss, uniform batches
    rwg   equal weight per group: (1/|G|) sum_g mean loss of group g,
          group-balanced batches (Idrissi et al., 2022)
    gdro  online Group DRO (Sagawa et al., 2020): q_g <- q_g exp(eta_q L_g),
          normalized; loss = sum_g q_g L_g; group-balanced batches.
          With eta_q = 0 this is exactly rwg (tested bit for bit).
    jtt   stage 2 of JTT (Liu et al., 2021): mean loss over a training set in
          which the error set of an ERM run at step T is upsampled lambda_up
          times. With lambda_up = 1 this is exactly erm (tested bit for bit).

The group-size adjustment C / sqrt(n_g) of Sagawa et al. is deliberately not
used, to keep tuning budgets equal across methods.
"""

from __future__ import annotations

import numpy as np
import torch

from .constants import N_GROUPS


def group_mean_losses(per_example: torch.Tensor, g: torch.Tensor, n_groups: int = N_GROUPS):
    """Mean loss of each group in the batch (0 for absent groups) and presence mask."""
    sums = torch.zeros(n_groups, device=per_example.device, dtype=per_example.dtype)
    sums = sums.scatter_add(0, g, per_example)
    counts = torch.bincount(g, minlength=n_groups).to(per_example.dtype)
    return sums / counts.clamp(min=1), counts > 0


class Objective:
    name = "base"
    sampler = "uniform"          # "uniform", "group_balanced" or "upsample"

    def __init__(self, train_y: np.ndarray, train_g: np.ndarray, **hparams):
        self.hparams = hparams
        self.multiplicity = None  # per-example repeat counts for the "upsample" sampler

    def loss(self, per_example, y, g, step):
        raise NotImplementedError

    def log_items(self) -> dict:
        return {}

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class ERM(Objective):
    name = "erm"

    def loss(self, per_example, y, g, step):
        return per_example.mean()


class GroupDRO(Objective):
    name = "gdro"
    sampler = "group_balanced"

    def __init__(self, train_y, train_g, eta_q: float = 0.01, **hparams):
        super().__init__(train_y, train_g, **hparams)
        self.eta_q = float(eta_q)
        self.q = torch.full((N_GROUPS,), 1.0 / N_GROUPS)

    def loss(self, per_example, y, g, step):
        means, present = group_mean_losses(per_example, g)
        q = self.q.to(per_example.device)
        if self.eta_q:
            # Exponentiated-gradient ascent on the group weights. Absent groups
            # have loss 0 here, so their weight is left unchanged before renormalizing.
            q = q * torch.exp(self.eta_q * means.detach() * present)
            q = q / q.sum()
        self.q = q
        return (q * means).sum()

    def log_items(self):
        return {f"q_g{k}": float(v) for k, v in enumerate(self.q.detach().cpu())}

    def state_dict(self):
        return {"q": self.q.detach().cpu().clone()}

    def load_state_dict(self, state):
        self.q = state["q"].clone()


class RWG(GroupDRO):
    """Group-balanced reweighting: Group DRO with the weights frozen at uniform."""

    name = "rwg"

    def __init__(self, train_y, train_g, **hparams):
        super().__init__(train_y, train_g, eta_q=0.0, **hparams)

    def log_items(self):
        return {}


class JTT(ERM):
    name = "jtt"
    sampler = "upsample"

    def __init__(self, train_y, train_g, error_mask: np.ndarray, jtt_lambda: int = 20, **hparams):
        super().__init__(train_y, train_g, **hparams)
        lam = int(jtt_lambda)
        if lam != jtt_lambda or lam < 1:
            raise ValueError(f"jtt_lambda must be a positive integer, got {jtt_lambda}")
        self.multiplicity = np.where(np.asarray(error_mask, bool), lam, 1).astype(np.int64)


OBJECTIVES = {"erm": ERM, "rwg": RWG, "gdro": GroupDRO, "jtt": JTT}

# Hyperparameters that belong to each method (and therefore enter its run hash).
METHOD_HPARAMS = {"erm": (), "rwg": (), "gdro": ("eta_q",), "jtt": ("jtt_T", "jtt_lambda")}


def make_objective(name: str, train_y: np.ndarray, train_g: np.ndarray, **hparams) -> Objective:
    if name not in OBJECTIVES:
        raise ValueError(f"unknown method {name!r}; implemented: {sorted(OBJECTIVES)}")
    return OBJECTIVES[name](train_y, train_g, **hparams)
