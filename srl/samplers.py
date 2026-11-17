"""Resumable minibatch index streams.

Training runs for a fixed number of gradient steps, not epochs, because
reweighting and upsampling change what an "epoch" means. The streams below
therefore yield an endless sequence of fixed-size batches; their state is
part of the checkpoint so a resumed run sees exactly the same batches.
"""

from __future__ import annotations

import numpy as np
import torch


class UniformStream:
    """Shuffled passes over range(n), concatenated: each example appears once
    per pass, and batches straddle pass boundaries so every batch is full."""

    def __init__(self, n: int, batch_size: int, generator: torch.Generator):
        if n <= 0:
            raise ValueError("cannot sample from an empty set")
        self.n, self.batch_size, self.gen = n, batch_size, generator
        self._perm = torch.empty(0, dtype=torch.long)
        self._pos = 0
        self.passes_started = 0

    def next_batch(self) -> torch.Tensor:
        parts, need = [], self.batch_size
        while need > 0:
            if self._pos >= len(self._perm):
                self._perm = torch.randperm(self.n, generator=self.gen)
                self._pos = 0
                self.passes_started += 1
            take = min(need, len(self._perm) - self._pos)
            parts.append(self._perm[self._pos:self._pos + take])
            self._pos += take
            need -= take
        return torch.cat(parts)

    def state_dict(self) -> dict:
        return {"perm": self._perm.clone(), "pos": self._pos, "passes_started": self.passes_started}

    def load_state_dict(self, state: dict) -> None:
        self._perm = state["perm"].clone()
        self._pos = int(state["pos"])
        self.passes_started = int(state["passes_started"])


class GroupBalancedStream:
    """Every batch holds batch_size / n_groups examples from each group, each
    group cycling through its own shuffled passes. This is a stratified
    version of the group-reweighted sampler used for Group DRO and
    reweighting; small groups are effectively upsampled. Not used by ERM."""

    def __init__(self, groups: np.ndarray, batch_size: int, generator: torch.Generator,
                 n_groups: int = 4):
        per_group, rem = divmod(batch_size, n_groups)
        if rem:
            raise ValueError(f"batch_size {batch_size} must be divisible by n_groups {n_groups}")
        groups = np.asarray(groups)
        self.members = [torch.from_numpy(np.flatnonzero(groups == g)) for g in range(n_groups)]
        empty = [g for g, m in enumerate(self.members) if len(m) == 0]
        if empty:
            raise ValueError(f"groups {empty} have no training examples")
        self.streams = [UniformStream(len(m), per_group, generator) for m in self.members]

    def next_batch(self) -> torch.Tensor:
        return torch.cat([m[s.next_batch()] for m, s in zip(self.members, self.streams)])

    def state_dict(self) -> dict:
        return {"streams": [s.state_dict() for s in self.streams]}

    def load_state_dict(self, state: dict) -> None:
        for s, st in zip(self.streams, state["streams"]):
            s.load_state_dict(st)


def make_stream(kind: str, n: int, groups: np.ndarray, batch_size: int,
                generator: torch.Generator):
    if kind == "uniform":
        return UniformStream(n, batch_size, generator)
    if kind == "group_balanced":
        return GroupBalancedStream(groups, batch_size, generator)
    raise ValueError(f"unknown sampler {kind!r}")
