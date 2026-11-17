"""Training objectives. Each method is an Objective that turns per-example
losses into the scalar that is backpropagated, declares which sampler it
needs, and carries any state (e.g. Group DRO's group weights) through
checkpoints.

Only ERM is implemented at this stage. RWG, Group DRO and JTT plug in here
without touching the training loop, which keeps every method on the same
data pipeline, optimizer, step budget and evaluation schedule.
"""

from __future__ import annotations

import numpy as np
import torch


class Objective:
    name = "base"
    sampler = "uniform"   # or "group_balanced"

    def __init__(self, train_y: np.ndarray, train_g: np.ndarray, **hparams):
        self.hparams = hparams

    def loss(self, per_example: torch.Tensor, y: torch.Tensor, g: torch.Tensor,
             step: int) -> torch.Tensor:
        raise NotImplementedError

    def log_items(self) -> dict:
        """Extra values recorded at every evaluation step."""
        return {}

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class ERM(Objective):
    """Empirical risk minimization: the plain average loss over the batch."""

    name = "erm"
    sampler = "uniform"

    def loss(self, per_example, y, g, step):
        return per_example.mean()


OBJECTIVES = {"erm": ERM}


def make_objective(name: str, train_y: np.ndarray, train_g: np.ndarray, **hparams) -> Objective:
    if name not in OBJECTIVES:
        raise ValueError(f"unknown method {name!r}; implemented: {sorted(OBJECTIVES)}")
    return OBJECTIVES[name](train_y, train_g, **hparams)
