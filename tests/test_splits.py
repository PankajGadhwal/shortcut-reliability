import numpy as np
import pytest

from srl.constants import EXPECTED_POOL_SIZE
from srl.data import (SplitSpec, SplitValidationError, counts_at_rho, group_counts, load_pool,
                      make_controlled_splits, resolve_sets, validate_controlled_splits,
                      verify_official_counts)
from srl.testing import fake_metadata


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    root = tmp_path_factory.mktemp("wb")
    fake_metadata().to_csv(root / "metadata.csv", index=False)
    return load_pool(root)


@pytest.fixture(scope="module")
def splits(pool):
    return make_controlled_splits(pool, SplitSpec())


def test_pool_is_canonical(pool):
    assert len(pool) == EXPECTED_POOL_SIZE
    assert np.all(np.diff(pool["img_id"].to_numpy()) > 0)
    assert set(pool["g"]) == {0, 1, 2, 3}
    verify_official_counts(pool)           # raises on mismatch


def test_official_count_mismatch_is_caught(pool):
    with pytest.raises(ValueError):
        verify_official_counts(pool.iloc[1:])


def test_counts_at_rho():
    assert counts_at_rho(0.95, 600).tolist() == [570, 30, 30, 570]
    assert counts_at_rho(0.99, 600).tolist() == [594, 6, 6, 594]
    assert counts_at_rho(0.5, 200).tolist() == [100, 100, 100, 100]


def test_default_design_is_valid_and_nested(pool, splits):
    summary = validate_controlled_splits(pool, splits)      # raises on any violation
    assert len(summary) == 25
    assert summary.loc[summary.rho == 0.99, "train_minority_per_class"].eq(6).all()
    s0 = splits["draws"]["0"]
    minority = lambda ids: set(np.asarray(ids)[np.isin(pool.set_index("img_id").loc[ids, "g"], [1, 2])])
    assert minority(s0["0.99"]["train"]) <= minority(s0["0.95"]["train"]) <= minority(s0["0.5"]["train"])


def test_fixed_sets_shared_and_balanced(pool, splits):
    assert group_counts(pool, splits["fixed"]["test"]).tolist() == [300] * 4
    assert group_counts(pool, splits["fixed"]["val_bal"]).tolist() == [100] * 4


def test_deterministic_and_seed_dependent(pool, splits):
    again = make_controlled_splits(pool, SplitSpec())
    assert again == splits
    other = make_controlled_splits(pool, SplitSpec(split_seed=1))
    assert other["fixed"]["test"] != splits["fixed"]["test"]
    assert splits["draws"]["0"]["0.95"]["train"] != splits["draws"]["1"]["0.95"]["train"]


def test_infeasible_spec_rejected(pool):
    with pytest.raises(ValueError, match="Infeasible"):
        make_controlled_splits(pool, SplitSpec(train_per_class=700))


def test_validator_catches_leak(pool, splits):
    broken = {**splits, "draws": {"0": {"0.95": dict(splits["draws"]["0"]["0.95"])}}}
    broken["draws"]["0"]["0.95"]["train"] = (broken["draws"]["0"]["0.95"]["train"][:-1]
                                             + [splits["fixed"]["test"][0]])
    with pytest.raises(SplitValidationError):
        validate_controlled_splits(pool, broken)


def test_resolve_sets(pool, splits):
    rs = resolve_sets("controlled", splits=splits, rho=0.9, data_seed=2)
    assert len(rs.train) == 1200 and set(rs.evals) == {"val_bal", "val_id", "test"}
    off = resolve_sets("official", pool=pool)
    assert (len(off.train), len(off.evals["val"]), len(off.evals["test"])) == (4795, 1199, 5794)
