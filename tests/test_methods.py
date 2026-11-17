"""Tests for RWG, Group DRO, JTT and DFR, including the exact-equivalence
checks promised in the plan: Group DRO with eta_q = 0 is RWG, and JTT with
lambda = 1 is ERM, bit for bit."""

import json

import numpy as np
import pytest
import torch

from srl.cache import build_cache
from srl.data import SplitSpec, load_pool, make_controlled_splits, save_splits
from srl.dfr import run_dfr
from srl.evaluate import evaluate_run
from srl.objectives import GroupDRO, group_mean_losses
from srl.samplers import UniformStream, UpsampleStream
from srl.testing import write_fake_dataset
from srl.train import TrainConfig, train

SMALL = SplitSpec(rhos=(0.5, 0.75), data_seeds=(0, 1), test_per_group=6, val_bal_per_group=4,
                  train_per_class=16, val_id_per_class=8, name="tiny")


# ---- unit tests --------------------------------------------------------------------

def test_group_mean_losses():
    losses = torch.tensor([1.0, 3.0, 5.0, 2.0])
    g = torch.tensor([0, 0, 2, 3])
    means, present = group_mean_losses(losses, g)
    assert means.tolist() == [2.0, 0.0, 5.0, 2.0]
    assert present.tolist() == [True, False, True, True]


def test_gdro_shifts_weight_to_the_worst_group():
    obj = GroupDRO(None, None, eta_q=0.5)
    g = torch.tensor([0, 1, 2, 3])
    for _ in range(20):
        obj.loss(torch.tensor([0.1, 0.1, 2.0, 0.1]), None, g, 0)
    assert obj.q.argmax().item() == 2 and torch.isclose(obj.q.sum(), torch.tensor(1.0))


def test_upsample_with_unit_multiplicity_equals_uniform():
    a = UniformStream(13, 4, torch.Generator().manual_seed(3))
    b = UpsampleStream(np.ones(13, dtype=int), 4, torch.Generator().manual_seed(3))
    assert all(torch.equal(a.next_batch(), b.next_batch()) for _ in range(10))


def test_upsample_counts():
    mult = np.array([1, 3, 1, 1])
    s = UpsampleStream(mult, 6, torch.Generator().manual_seed(0))
    seen = torch.cat([s.next_batch() for _ in range(5)])            # exactly 5 passes of 6
    assert torch.bincount(seen, minlength=4).tolist() == [5, 15, 5, 5]


# ---- end to end -------------------------------------------------------------------

@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("methods")
    counts = {"train": (14, 6, 6, 14), "val": (8, 8, 8, 8), "test": (8, 16, 16, 8)}
    root = write_fake_dataset(tmp / "wb", counts)
    pool = load_pool(root)
    build_cache(pool, root, tmp / "cache", size=36, workers=1)
    save_splits(tmp / "splits.json", make_controlled_splits(pool, SMALL))
    return tmp


def cfg(env, **kw):
    base = dict(cache_dir=str(env / "cache"), splits_path=str(env / "splits.json"), rho=0.75,
                data_seed=1, arch="resnet18", pretrained=False, steps=12, batch_size=8, crop=32,
                eval_every=4, dense_eval_every=2, dense_eval_until=4, ckpt_every=4,
                eval_batch_size=16, out_dir=str(env / "runs"), amp=False)
    base.update(kw)
    return TrainConfig(**base)


def logits(run, name="val_bal"):
    folder = "sealed" if name == "test" else "logits"
    return np.load(run / folder / f"{name}.npy")


def test_gdro_with_zero_step_size_is_rwg(env):
    rwg = train(cfg(env, method="rwg"))
    gdro0 = train(cfg(env, method="gdro", eta_q=0.0))
    assert rwg != gdro0
    for name in ("train", "val_bal", "test"):
        assert np.array_equal(logits(rwg, name), logits(gdro0, name)), name


def test_gdro_runs_logs_weights_and_resumes_exactly(env):
    full = train(cfg(env, method="gdro", eta_q=0.5))
    log = open(full / "train_log.csv").read()
    assert "q_g0" in log and "q_g3" in log
    paused = train(cfg(env, method="gdro", eta_q=0.5, out_dir=str(env / "runs_resume"),
                       stop_after_step=6))
    assert not (paused / "DONE").exists()
    resumed = train(cfg(env, method="gdro", eta_q=0.5, out_dir=str(env / "runs_resume")))
    assert np.array_equal(logits(full), logits(resumed))


def test_jtt_needs_its_erm_run(env):
    with pytest.raises(FileNotFoundError, match="JTT needs"):
        train(cfg(env, method="jtt", jtt_T=4, out_dir=str(env / "runs_empty")))


def test_jtt_with_lambda_one_is_erm_and_audit_is_written(env):
    erm = train(cfg(env, method="erm"))
    jtt1 = train(cfg(env, method="jtt", jtt_T=4, jtt_lambda=1))
    for name in ("train", "val_bal", "test"):
        assert np.array_equal(logits(erm, name), logits(jtt1, name)), name
    jtt = train(cfg(env, method="jtt", jtt_T=4, jtt_lambda=5))
    audit = json.load(open(jtt / "jtt_error_set.json"))
    assert audit["reference_run"] == erm.name
    assert audit["error_set_size"] == len(audit["error_indices"])
    with pytest.raises(ValueError, match="not a logged step"):
        train(cfg(env, method="jtt", jtt_T=3))


def test_erm_run_name_is_unaffected_by_method_hyperparameters(env):
    a = train(cfg(env, method="erm"))
    b = train(cfg(env, method="erm", eta_q=0.9, jtt_T=8, jtt_lambda=7))
    assert a == b


def test_dfr_end_to_end(env):
    erm = train(cfg(env, method="erm"))
    out = run_dfr(erm, c_grid=(1.0, 0.1), n_avg=2)
    conf = json.load(open(out / "config.json"))
    assert conf["method"] == "dfr" and conf["protocols"] == ["S1"]
    assert logits(out, "val_bal").shape == (1, 16, 2) and logits(out, "test").shape == (1, 24, 2)
    rows = evaluate_run(out, unseal=True)
    assert [r["protocol"] for r in rows] == ["S1"] and "test_wga" in rows[0]
    assert run_dfr(erm, c_grid=(1.0, 0.1), n_avg=2) == out            # rerun is skipped
