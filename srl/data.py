"""Waterbirds pool loading, verification against the official counts, and the
controlled re-splitting used for the correlation-strength study.

Design (project plan, Section 4)
--------------------------------
* All 11,788 Waterbirds images form one pool; each has a group label g=(y, a).
* Fixed once (split_seed): a group-balanced test set and a group-balanced
  validation set, val_bal. These never change across rho or seeds, so every
  method and every rho is scored on exactly the same images.
* For each data seed and each correlation strength rho: a class-balanced
  training set and an in-distribution validation set, val_id, both drawn at
  correlation rho = P(background agrees with class | class).
* Nested sampling: per seed, each group's remaining pool is shuffled ONCE.
  Training examples are read from the front of that order and val_id
  examples from the back. Consequences:
    - for a fixed seed, training sets at different rho are nested (the 6
      minority images per class at rho=0.99 are among the 30 at rho=0.95),
      so differences across rho reflect rho, not which images were drawn;
    - train and val_id can never overlap at any rho.

Splits are stored as JSON lists of img_id and should be committed to git:
splits are data, not code, and regenerating them on another machine or
NumPy version is not guaranteed to reproduce them.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (EXPECTED_OFFICIAL_COUNTS, GROUP_NAMES, N_GROUPS,
                        OFFICIAL_SPLIT_CODES)
from .utils import load_json, rho_key, save_json, sha256_file

REQUIRED_COLUMNS = ("img_id", "img_filename", "y", "split", "place")


class SplitValidationError(ValueError):
    """Raised when a split violates a design invariant (counts, disjointness, nesting)."""


# ----------------------------------------------------------------------------
# Pool
# ----------------------------------------------------------------------------

def load_pool(data_root: str | Path) -> pd.DataFrame:
    """Read metadata.csv from the extracted Waterbirds folder.

    Returns one row per image, sorted by img_id (the canonical order used by
    the image cache), with columns img_id, img_filename, y, a, g,
    official_split and, if present, place_filename.
    """
    meta_path = Path(data_root) / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Point data_root at the extracted "
            "'waterbird_complete95_forest2water2' folder.")
    meta = pd.read_csv(meta_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in meta.columns]
    if missing:
        raise ValueError(f"metadata.csv lacks columns {missing}; found {list(meta.columns)}")

    df = pd.DataFrame({
        "img_id": meta["img_id"].astype(np.int64),
        "img_filename": meta["img_filename"].astype(str),
        "y": meta["y"].astype(np.int64),
        "a": meta["place"].astype(np.int64),
        "official_split": meta["split"].astype(np.int64),
    })
    if "place_filename" in meta.columns:
        df["place_filename"] = meta["place_filename"].astype(str)
    if not df["img_id"].is_unique:
        raise ValueError("img_id is not unique in metadata.csv")
    for col, allowed in (("y", {0, 1}), ("a", {0, 1}), ("official_split", {0, 1, 2})):
        unexpected = set(df[col].unique().tolist()) - allowed
        if unexpected:
            raise ValueError(f"column {col!r} has unexpected values {sorted(unexpected)}")
    df["g"] = 2 * df["y"] + df["a"]
    return df.sort_values("img_id").reset_index(drop=True)


def _group_lookup(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df["g"].to_numpy(), index=df["img_id"].to_numpy())


def group_counts(df: pd.DataFrame, img_ids) -> np.ndarray:
    """Per-group counts of a set of img_ids."""
    g = _group_lookup(df).reindex(np.asarray(img_ids))
    if g.isna().any():
        raise KeyError(f"{int(g.isna().sum())} img_ids not found in the pool")
    return np.bincount(g.to_numpy().astype(np.int64), minlength=N_GROUPS)


def official_count_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, code in OFFICIAL_SPLIT_CODES.items():
        observed = np.bincount(df.loc[df["official_split"] == code, "g"], minlength=N_GROUPS)
        for g in range(N_GROUPS):
            rows.append({"split": name, "group": GROUP_NAMES[g],
                         "observed": int(observed[g]),
                         "expected": EXPECTED_OFFICIAL_COUNTS[name][g]})
    table = pd.DataFrame(rows)
    table["match"] = table["observed"] == table["expected"]
    return table


def verify_official_counts(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Compare against the official release. A mismatch means a corrupted or
    different download, and every number downstream would be suspect."""
    table = official_count_table(df)
    if not table["match"].all():
        msg = ("Waterbirds counts differ from the official release:\n"
               + table[~table["match"]].to_string(index=False))
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)
    return table


# ----------------------------------------------------------------------------
# Controlled splits
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitSpec:
    rhos: tuple = (0.5, 0.8, 0.9, 0.95, 0.99)
    data_seeds: tuple = (0, 1, 2, 3, 4)
    split_seed: int = 0
    test_per_group: int = 300
    val_bal_per_group: int = 100
    train_per_class: int = 600
    val_id_per_class: int = 200
    name: str = "controlled_v1"


def counts_at_rho(rho: float, per_class: int) -> np.ndarray:
    """Per-group counts (ordered by g) for a class-balanced set at correlation rho."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [0, 1], got {rho}")
    n_agree = int(round(rho * per_class))
    n_disagree = per_class - n_agree
    return np.array([n_agree, n_disagree, n_disagree, n_agree], dtype=np.int64)


def make_controlled_splits(df: pd.DataFrame, spec: SplitSpec = SplitSpec()) -> dict:
    rng = np.random.default_rng(spec.split_seed)
    t, v = spec.test_per_group, spec.val_bal_per_group
    fixed = {"test": [], "val_bal": []}
    remaining = {}
    for g in range(N_GROUPS):
        ids = rng.permutation(np.sort(df.loc[df["g"] == g, "img_id"].to_numpy()))
        if t + v > len(ids):
            raise ValueError(f"group {GROUP_NAMES[g]} has {len(ids)} images; "
                             f"the fixed sets need {t + v}")
        fixed["test"].append(ids[:t])
        fixed["val_bal"].append(ids[t:t + v])
        remaining[g] = ids[t + v:]

    # Feasibility: the longest training prefix and the longest val_id suffix,
    # over all rho, must fit in each group's remaining pool without touching.
    need_train = np.max([counts_at_rho(r, spec.train_per_class) for r in spec.rhos], axis=0)
    need_val = np.max([counts_at_rho(r, spec.val_id_per_class) for r in spec.rhos], axis=0)
    for g in range(N_GROUPS):
        if need_train[g] + need_val[g] > len(remaining[g]):
            raise ValueError(
                f"Infeasible spec: group {GROUP_NAMES[g]} has {len(remaining[g])} images "
                f"after the fixed sets but needs {need_train[g]} (train) + "
                f"{need_val[g]} (val_id).")

    draws = {}
    for s in spec.data_seeds:
        rng_s = np.random.default_rng([spec.split_seed, int(s)])
        order = {g: rng_s.permutation(remaining[g]) for g in range(N_GROUPS)}
        draws[str(s)] = {}
        for rho in spec.rhos:
            ct = counts_at_rho(rho, spec.train_per_class)
            cv = counts_at_rho(rho, spec.val_id_per_class)
            train = np.concatenate([order[g][:ct[g]] for g in range(N_GROUPS)])
            val_id = np.concatenate([order[g][len(order[g]) - cv[g]:] for g in range(N_GROUPS)])
            draws[str(s)][rho_key(rho)] = {"train": sorted(train.tolist()),
                                           "val_id": sorted(val_id.tolist())}

    return {
        "format_version": 1,
        "spec": asdict(spec),
        "pool_size": int(len(df)),
        "fixed": {k: sorted(np.concatenate(parts).tolist()) for k, parts in fixed.items()},
        "draws": draws,
    }


def _realized_rho(counts: np.ndarray) -> tuple[float, float]:
    """Fraction of background-agreeing examples within each class."""
    land = counts[0] / max(counts[0] + counts[1], 1)
    water = counts[3] / max(counts[2] + counts[3], 1)
    return float(land), float(water)


def validate_controlled_splits(df: pd.DataFrame, splits: dict) -> pd.DataFrame:
    """Check every design invariant; raise SplitValidationError on violation.
    Returns one summary row per (seed, rho)."""
    spec = splits["spec"]
    test, val_bal = set(splits["fixed"]["test"]), set(splits["fixed"]["val_bal"])
    if test & val_bal:
        raise SplitValidationError("test and val_bal overlap")
    for name, ids, per_group in (("test", test, spec["test_per_group"]),
                                 ("val_bal", val_bal, spec["val_bal_per_group"])):
        c = group_counts(df, sorted(ids))
        if not np.all(c == per_group):
            raise SplitValidationError(f"{name} group counts {c.tolist()} != {per_group} each")

    rows = []
    for seed, by_rho in splits["draws"].items():
        nested_train = {g: [] for g in range(N_GROUPS)}
        nested_val = {g: [] for g in range(N_GROUPS)}
        g_of = _group_lookup(df)
        for rk, d in by_rho.items():
            rho = float(rk)
            tr, vi = set(d["train"]), set(d["val_id"])
            for a_name, a, b_name, b in (("train", tr, "val_id", vi), ("train", tr, "test", test),
                                         ("train", tr, "val_bal", val_bal),
                                         ("val_id", vi, "test", test),
                                         ("val_id", vi, "val_bal", val_bal)):
                if a & b:
                    raise SplitValidationError(
                        f"seed {seed}, rho {rk}: {a_name} and {b_name} share {len(a & b)} images")
            ct, cv = group_counts(df, d["train"]), group_counts(df, d["val_id"])
            if not np.array_equal(ct, counts_at_rho(rho, spec["train_per_class"])):
                raise SplitValidationError(f"seed {seed}, rho {rk}: train counts {ct.tolist()}")
            if not np.array_equal(cv, counts_at_rho(rho, spec["val_id_per_class"])):
                raise SplitValidationError(f"seed {seed}, rho {rk}: val_id counts {cv.tolist()}")
            tr_g = g_of.reindex(d["train"]).to_numpy()
            vi_g = g_of.reindex(d["val_id"]).to_numpy()
            for g in range(N_GROUPS):
                nested_train[g].append(set(np.asarray(d["train"])[tr_g == g].tolist()))
                nested_val[g].append(set(np.asarray(d["val_id"])[vi_g == g].tolist()))
            land, water = _realized_rho(ct)
            rows.append({"seed": int(seed), "rho": rho,
                         **{f"train_{GROUP_NAMES[g]}": int(ct[g]) for g in range(N_GROUPS)},
                         "train_minority_per_class": int(ct[1]),
                         "realized_rho_landbirds": land, "realized_rho_waterbirds": water,
                         **{f"val_id_{GROUP_NAMES[g]}": int(cv[g]) for g in range(N_GROUPS)}})
        for kind, nested in (("train", nested_train), ("val_id", nested_val)):
            for g, sets in nested.items():
                ordered = sorted(sets, key=len)
                for small, large in zip(ordered, ordered[1:]):
                    if not small <= large:
                        raise SplitValidationError(
                            f"seed {seed}: {kind} sets of group {GROUP_NAMES[g]} are not nested")
    return pd.DataFrame(rows).sort_values(["seed", "rho"]).reset_index(drop=True)


def background_overlap(df: pd.DataFrame, ids_a, ids_b) -> int | None:
    """Number of Places background images shared between two sets of img_ids
    (None if metadata.csv has no place_filename column)."""
    if "place_filename" not in df.columns:
        return None
    pf = pd.Series(df["place_filename"].to_numpy(), index=df["img_id"].to_numpy())
    return len(set(pf.reindex(np.asarray(ids_a)).tolist()) & set(pf.reindex(np.asarray(ids_b)).tolist()))


def save_splits(path: str | Path, splits: dict) -> str:
    save_json(path, splits, compact=True)
    return sha256_file(path)


def load_splits(path: str | Path) -> tuple[dict, str]:
    return load_json(path), sha256_file(path)


# ----------------------------------------------------------------------------
# Resolving the sets for one training run
# ----------------------------------------------------------------------------

@dataclass
class RunSets:
    train: np.ndarray            # img_ids
    evals: dict                  # name -> img_ids; 'test' is sealed
    description: str


def resolve_sets(split: str, *, splits: dict | None = None, rho: float | None = None,
                 data_seed: int = 0, pool: pd.DataFrame | None = None) -> RunSets:
    if split == "controlled":
        if splits is None or rho is None:
            raise ValueError("controlled split needs a splits file and rho")
        seed_key, rk = str(data_seed), rho_key(rho)
        if seed_key not in splits["draws"]:
            raise KeyError(f"data_seed {data_seed} not in splits (have {list(splits['draws'])})")
        if rk not in splits["draws"][seed_key]:
            raise KeyError(f"rho {rk} not in splits (have {list(splits['draws'][seed_key])})")
        d = splits["draws"][seed_key][rk]
        return RunSets(
            train=np.asarray(d["train"], dtype=np.int64),
            evals={"val_bal": np.asarray(splits["fixed"]["val_bal"], dtype=np.int64),
                   "val_id": np.asarray(d["val_id"], dtype=np.int64),
                   "test": np.asarray(splits["fixed"]["test"], dtype=np.int64)},
            description=f"controlled rho={rk} data_seed={data_seed}")
    if split == "official":
        if pool is None:
            raise ValueError("official split needs the pool/index table")
        ids = pool["img_id"].to_numpy()
        code = pool["official_split"].to_numpy()
        return RunSets(train=ids[code == 0], evals={"val": ids[code == 1], "test": ids[code == 2]},
                       description="official Waterbirds split")
    raise ValueError(f"unknown split {split!r}")
